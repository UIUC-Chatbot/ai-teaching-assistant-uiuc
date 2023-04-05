import os
import sys

# set your OpenAI API key here
# os.environ["OPENAI_API_KEY"] = ""
os.environ["TRANSFORMERS_CACHE"] = "/mnt/project/chatbotai/huggingface_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "~/.cache"

sys.path.append("../human_data_review")

import json
from datetime import datetime
from typing import Any, List, Tuple

import evaluate
import numpy as np
# import TA_gradio_ux
import pandas as pd
import pinecone
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.evaluation.qa import QAEvalChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpu_memory_utils import (get_device_with_most_free_memory, get_free_memory_dict, get_gpu_ids_with_sufficient_memory)
from main import TA_Pipeline

load_dotenv(dotenv_path='/mnt/project/chatbotai/huggingface_cache/internal_api_keys.env', override=True)

# GLOBALS
NUM_OF_DATAPOINTS_TO_EVALUATE = 3

# "prefix_begin": "<|prefix_begin|>"
# "prefix_end": "<|prefix_end|>"
OPEN_ASSISTANT_PROMPTS_TO_TEST = [
    PromptTemplate(
        template=
        '''<prefix>You are a helpful and precise assistant for answering factual questions about Electrical Engineering. If it's helpful, consider using the provided context to help with your answer.</prefix>
        <|prompter|>Context: {context}
        Question: {question}<|endoftext|><|assistant|>
        ''',
        input_variables=["question", "context"],
    ),
    PromptTemplate(
        template=
        '''<|prefix_begin|>You are a helpful and precise assistant for answering factual questions about Electrical Engineering. If it's helpful, consider using the provided context to help with your answer.<|prefix_end|>
        <|prompter|>Context: {context}
        Question: {question}
        Answer:<|endoftext|><|assistant|>
        ''',
        input_variables=["question", "context"],
    ),
    PromptTemplate(
        template=
        '''<prefix>You are a helpful and precise assistant for answering factual questions about Electrical Engineering. If it's helpful, consider using the provided context to help with your answer.</prefix>
        <|prompter|>Context: {context}
        Please answer this question as accuratly as possible and with as much detail as possible.
        Question: {question}
        Answer:<|endoftext|><|assistant|>
        ''',
        input_variables=["question", "context"],
    )
]


class Evaluator():

  def __init__(self) -> None:
    # self.ta_pipeline = TA_Pipeline(dont_load_any_cuda=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.model = AutoModelForCausalLM.from_pretrained("OpenAssistant/oasst-sft-1-pythia-12b",
                                                      device_map="sequential",
                                                      max_memory=get_free_memory_dict(leave_extra_memory_unused_GiB=5,
                                                                                      leave_extra_memory_unused_gpu0_GiB=6))
    self.tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-1-pythia-12b")

    self.vectorstore = None
    self._load_pinecone_vectorstore()

  def _load_pinecone_vectorstore(self,):
    model_name = "intfloat/e5-large"  # best text embedding model. 1024 dims.
    pincecone_index = pinecone.Index(os.environ['PINECONE_INDEX_NAME'])
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # pinecone.init(api_key=os.environ['PINECONE_API_KEY_NEW_ACCT'], environment="us-east4-gcp")
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENVIRONMENT'])
    self.vectorstore = Pinecone(index=pincecone_index, embedding_function=embeddings.embed_query, text_key="text")

  def retrieve_contexts_from_pinecone(self, user_question: str, topk: int = 3) -> List[str]:
    ''' 
    Invoke Pinecone for vector search. These vector databases are created in the notebook `data_formatting_patel.ipynb` and `data_formatting_student_notes.ipynb`.
    Returns a list of LangChain Documents. They have properties: `doc.page_content`: str, doc.metadata['page_number']: int, doc.metadata['textbook_name']: str.
    '''
    print("WARNING USING STATIC CONTEXT")
    return ["The finite state machine is a nice model of ECE!"]
    # similarity search
    top_context_list = self.vectorstore.similarity_search(user_question, k=topk)

    # add the source info to the bottom of the context.
    top_context_metadata = [f"Source: page {doc.metadata['page_number']} in {doc.metadata['textbook_name']}" for doc in top_context_list]
    relevant_context_list = [f"{text.page_content}. {meta}" for text, meta in zip(top_context_list, top_context_metadata)]
    return relevant_context_list

  def get_open_assistant_prompt(self, prompt_template: PromptTemplate, input_question: str):
    """
    Args: prompt_tempate: the template of the prompt
            question: the question
    Returns: the prompt for OpenAssistant
    """
    # call pinecone for contexts
    context = self.retrieve_contexts_from_pinecone(input_question, topk=1)[0]
    prompt = prompt_template.format(question=input_question, context=context)
    return prompt

  def open_assistant(self, prompt_template: PromptTemplate, input_question: str) -> str:
    """
    Args: input user's question
    Returns: output OpenAssistant generated answer
    """

    prompt = self.get_open_assistant_prompt(prompt_template, input_question)
    print("PROPT as sent to model:", prompt)

    inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")  # always 0 or .generate()
    tokens = self.model.generate(**inputs, max_new_tokens=500, typical_p=0.2, temperature=0.6, pad_token_id=self.tokenizer.eos_token_id)
    output = self.tokenizer.decode(tokens[0])
    output = output.split('<|assistant|>')[1].split('<|endoftext|>')[0]
    return output

  def langchain_grader(self, eval_dataset):
    """
      Args: evaluation set path: dataset path for GPT-3 evaluation

      Returns: output TWO result files
      1. Generate a .json file which contains {question, original answer, new generated answer, GPT-3 grade}
      2. Generate a new .json eval set, which contains {question, new better answer}
      
      Change each file path you would like to evaluate or generate
      """

    # eval_dataset = json.load(open(eval_set_path, 'r'))
    eval_qa = []
    best_generated_answer = []
    # Create a prompt for generate GPT-3 grade label
    _PROMPT_TEMPLATE = """You are an expert professor specialized in evaluating students' new answers comparing to the ground truth answers given the same questions.
                          You are referring the following question:
                          {query}
                          Here is the ground truth answer:
                          {answer}
                          You are evaluating the following new answer:
                          {result}
                          Do you think the new answer is better than the ground truth answer? Label as "Better" or "Worse".
                          """
    PROMPT = PromptTemplate(input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE)

    # Process Huggingface Eval Dataset
    eval_dataframe = pd.DataFrame()
    eval_dataframe['prompt'] = eval_dataset['train']['prompt'][:NUM_OF_DATAPOINTS_TO_EVALUATE]
    eval_dataframe['completion'] = eval_dataset['train']['completion'][:NUM_OF_DATAPOINTS_TO_EVALUATE]

    for prompt_template in OPEN_ASSISTANT_PROMPTS_TO_TEST:
      for question, ans in zip(eval_dataframe['prompt'], eval_dataframe['completion']):
        temp_q_dict = {}
        temp_new_answer_dict = {}
        temp_q_dict['question'] = question
        temp_q_dict['answer'] = ans

        # generate answer using OpenAssistant
        generated_answer = self.open_assistant(prompt_template, question)

        # previous T5 question answer pipeline
        # generated_answers, _ = self.ta_pipeline.question_answer(question, "")
        # temp_new_answer_dict['text'] = generated_answers["Answer"].head(1).values

        temp_new_answer_dict['text'] = generated_answer
        eval_qa.append(temp_q_dict)
        best_generated_answer.append(temp_new_answer_dict)

    # Load LangChain Evaluation pipeline
    eval_model = OpenAI(temperature=0)
    evalchain = QAEvalChain.from_llm(llm=eval_model, prompt=PROMPT)
    # Grade the new model generated answer compared to the original one
    grader = evalchain.evaluate(eval_qa, best_generated_answer, question_key="question", answer_key="answer", prediction_key="text")

    # Add the new evaluation results to a new evaluation set (w/ two answers version)
    # and the original evaluation set (cover the worse answers)
    new_eval_set = []
    # updated_eval_set = []
    for i, (q, a) in enumerate(zip(eval_dataframe['prompt'], eval_dataframe['completion'])):
      new_generated_answer = best_generated_answer[i]['text']
      # new_generated_answer = best_generated_answer[i]['text'][0]
      grade_label = grader[i]['text'].replace('\n', '')
      temp_row = {}
      temp_row['Question'] = q
      temp_row['Original-Ground-Truth'] = a
      temp_row['Chatbot-Generated-Answer'] = new_generated_answer
      temp_row['GPT-3-Evaluation'] = grade_label
      new_eval_set.append(temp_row)

    # Write the new evaluation data to the JSON file
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as a string
    timestamp = now.strftime("%Y-%m-%d_%H-%M")
    # Create a file name with the date and time as a suffix
    file_name = "/home/zhiweny2/chatbotai/jerome/human_data_review/" + "gpt3_graded_set_use_OpenAssistant_" + timestamp + ".json"
    # Write the new evaluation data (w/ two compared answers verision) to the JSON file
    # The format of the JSON file includes: question, original answer, chatbot generated answer, GPT-3 evaluation label
    # Change the path you want to save this file for human comparision only
    with open(file_name, 'w', encoding='utf-8') as f:
      json.dump(new_eval_set, f, ensure_ascii=False, indent=4)
    # Write the updated evaluation data to the JSON file
    # Change the path you want to save this updated eval set for further evaluation
    # with open('/home/zhiweny2/chatbotai/jerome/human_data_review/new_eval_set.json', 'w', encoding='utf-8') as f:
    #     json.dump(updated_eval_set, f, ensure_ascii=False, indent=4)

  def rouge_n_bleu_score(self, eval_dataset):
    """
      Args:user_question (str): questions from the human filtered eval set

      Returns: the evaluation score for each user's question compared to the human labeled answers
      
      overall_rouge_score: the average RougeL f1 score for all the questions in eval set
      overall_bleu_score: the average Bleu1 score for all the questions in eval set
      """
    # set the data points of evaluation to 30
    NUM_OF_DATAPOINTS_TO_EVALUATE = 30
    # eval_dataset = json.load(open(eval_set_path, 'r'))
    bleu_metric = evaluate.load('bleu')
    rouge = Rouge()
    rouge_score_list, bleu_score_list = [], []

    eval_dataframe = pd.DataFrame()
    eval_dataframe['prompt'] = eval_dataset['train']['Question'][:NUM_OF_DATAPOINTS_TO_EVALUATE]
    eval_dataframe['completion'] = eval_dataset['train']['Chosen'][:NUM_OF_DATAPOINTS_TO_EVALUATE]

    for i, (question, answer) in enumerate(zip(eval_dataframe['prompt'], eval_dataframe['completion'])):
      generated_answers, _ = self.ta_pipeline.question_answer(question, "")
      best_generated_answer = generated_answers["Answer"].head(1).values
      best_generated_answer = best_generated_answer[0].replace("<s>", "")
      # rouge score
      rouge_scores = rouge.get_scores(best_generated_answer, answer)
      rougel_f_score = rouge_scores[0]['rouge-l']['f']
      rouge_score_list.append(rougel_f_score)
      # bleu score
      bleu_scores = bleu_metric.compute(predictions=[best_generated_answer.split(' ')], references=[[answer.split(' ')]])
      bleu_1_socre = bleu_scores['precisions'][0]
      bleu_score_list.append(bleu_1_socre)
    overall_rouge_score = np.mean(rouge_score_list)
    overall_bleu_score = np.mean(bleu_score_list)
    return overall_rouge_score, overall_bleu_score


def main():
  eval_dataset = load_dataset("kastan/rlhf-qa-conditional-generation-v2")
  evaluator = Evaluator()
  evaluator.langchain_grader(eval_dataset)


if __name__ == '__main__':
  main()

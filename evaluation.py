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

import datasets
import evaluate
import numpy as np
# import TA_gradio_ux
import pandas as pd
import pinecone
import torch
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.evaluation.qa import QAEvalChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from rouge import Rouge
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpu_memory_utils import (get_device_with_most_free_memory, get_free_memory_dict, get_gpu_ids_with_sufficient_memory)

load_dotenv(dotenv_path='/mnt/project/chatbotai/huggingface_cache/internal_api_keys.env', override=True)

# GLOBALS
NUM_OF_DATAPOINTS_TO_EVALUATE = 100

# TODO: Try better prompts. See prompting.py


OPEN_ASSISTANT_PROMPTS_TO_TEST = [
    # Few shot prompt
    PromptTemplate(
        template=
        '''<|prefix_begin|>Generate an objective and logical answer to this question, based on the context. The answer should be short, to-the-point while being substantial as per a freshmen-level language. Give examples.<|prefix_end|>
<|prompter|>Context: LC-3 ISA A.1 Overview The instruction set architecture (ISA) of the LC-3 is defined as follows: Memory address space 16 bits, corresponding to 216 locations, each containing one word (16 bits). Addresses are numbered from 0 (i.e., x0000) to 65,535 (i.e., xFFFF). Addresses are used to identify memory locations and memory-mapped I/O device registers.

Question: What is LC-3?
Answer: The instruction set architecture (ISA) of the LC-3 is a 16-bit ISA. The architecture specifies the types of instructions and their addressing modes, as well as the size and layout of the memory space.
The instruction set includes basic arithmetic and logical operations, such as addition, subtraction, multiplication, and division. It also includes bitwise operations, including logical AND, logical OR, and bit shift.
The ISA also provides for memory access, including reads and writes. Memory accesses are performed using memory addresses, which are specified in terms of the base address and the displacement. The base address specifies the starting location of the memory word, while the displacement specifies the number of bits to be retrieved or written from the memory.

Context: {context}

Question: {question}
Answer: <|endoftext|><|assistant|>''',
        input_variables=["question", "context"],
    ),
    # Answer based on below context. Simple prompt good with GPT-3/4.
    PromptTemplate(
        template=
        '''<|prefix_begin|>Answer the question based on the context below. If the question cannot be answered using the provided context, answer the best your can or answer with "I'm not sure, but..." with your best guess.<|prefix_end|>
<|prompter|>Context: {context}. 
Question: {question}<|endoftext|><|assistant|>''',
        input_variables=["question", "context"],
    ),

    # let's think step by step
    PromptTemplate(
        template=
        '''<prefix>Generate an objective and logical answer to this question, based on the context. The answer should be short, to-the-point while being substantial as per a freshmen-level language. Do not include any irrelevant information. Give examples. Let's think step by step.</prefix>
<|prompter|>Context: {context}. 
Please answer this question accuratly with detail and an example.
Question: {question}<|endoftext|><|assistant|>''',
        input_variables=["question", "context"],
    ),
    # previous best
    PromptTemplate(
        template=
        '''<prefix>Generate an objective and logical answer to this question, based on the context. The answer should be short, to-the-point while being substantial as per a freshmen-level language. Do not include any irrelevant information. Give examples.</prefix>
<|prompter|>Context: {context}. 
Please answer this question accuratly with detail and an example.
Question: {question}<|endoftext|><|assistant|>''',
        input_variables=["question", "context"],
    )
]


class Evaluator():

  def __init__(self) -> None:
    self.model = AutoModelForCausalLM.from_pretrained("OpenAssistant/oasst-sft-1-pythia-12b",
                                                      device_map="sequential",
                                                      max_memory=get_free_memory_dict(leave_extra_memory_unused_GiB=5,
                                                                                      leave_extra_memory_unused_gpu0_GiB=6))
    self.tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-1-pythia-12b")

    self.vectorstore = None
    self._load_pinecone_vectorstore()

  def _load_pinecone_vectorstore(self):
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENVIRONMENT'])
    pincecone_index = pinecone.Index(os.environ['PINECONE_INDEX_NAME'])
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large")  # best text embedding model. 1024 dims.
    self.vectorstore = Pinecone(index=pincecone_index, embedding_function=embeddings.embed_query, text_key="text")

  def retrieve_contexts_from_pinecone(self, user_question: str, topk: int = 3) -> List[str]:
    ''' 
    Call Pinecone for relevant document contexts.
    
    Args: prompt_tempate: the template of the prompt
            question: the question
    Returns: List of strings, each is a context. 
    
    These vector databases are created in the notebook `data_formatting_patel.ipynb` and `data_formatting_student_notes.ipynb`.
    '''
    top_context_list = self.vectorstore.similarity_search(user_question, k=topk)

    # add the source info to the bottom of the context.
    # top_context_metadata = [f"Source: page {doc.metadata['page_number']} in {doc.metadata['textbook_name']}" for doc in top_context_list]
    # relevant_context_list = [f"{text.page_content}. {meta}" for text, meta in zip(top_context_list, top_context_metadata)]
    return top_context_list

  def get_open_assistant_prompt(self, prompt_template: PromptTemplate, input_question: str):
    """
    Args: prompt_tempate: the template of the prompt
            question: the question
    Returns: the prompt for OpenAssistant
    """
    context = self.retrieve_contexts_from_pinecone(input_question, topk=1)[0]
    prompt = prompt_template.format(question=input_question, context=context)
    return prompt

  def open_assistant(self, prompt_template: PromptTemplate, input_question: str) -> str:
    """
    Args: input user's question
    Returns: output OpenAssistant generated answer
    """

    prompt = self.get_open_assistant_prompt(prompt_template, input_question).replace('\n', '')
    # print("PROMPT as sent to model:", prompt)

    inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")  # always 0 for .generate()
    tokens = self.model.generate(**inputs, max_new_tokens=500, typical_p=0.2, temperature=0.6, pad_token_id=self.tokenizer.eos_token_id)
    temp_output = self.tokenizer.decode(tokens[0])
    output = temp_output.split('<|assistant|>')[1].split('<|endoftext|>')[0]
    return output

  def main_eval_loop(self, eval_dataset: datasets.Dataset):
    '''
    Main driver of evaluation loop. Currently just evalauting OPEN_ASSISTANT_PROMPTS_TO_TEST.
    Param: dataset 
    '''
    eval_dataframe = pd.DataFrame()
    eval_dataframe['prompt'] = eval_dataset['prompt'][:NUM_OF_DATAPOINTS_TO_EVALUATE]
    eval_dataframe['completion'] = eval_dataset['completion'][:NUM_OF_DATAPOINTS_TO_EVALUATE]

    # 1 eval per experimental condition.
    for i, prompt_template in enumerate(OPEN_ASSISTANT_PROMPTS_TO_TEST):
      self.langchain_grader(eval_dataframe=eval_dataframe, prompt_template=prompt_template, eval_name=f'OAsst_prompt_{i}')

  def langchain_grader(self, eval_dataframe: pd.DataFrame, prompt_template: PromptTemplate, eval_name: str = 'default_eval') -> None:
    """
      Args: evaluation set path: dataset path for GPT-3 evaluation

      Returns: output TWO result files
      1. Generate a .json file which contains {question, original answer, new generated answer, GPT-3 grade}
      2. Generate a new .json eval set, which contains {question, new better answer}
      
      Change each file path you would like to evaluate or generate
      """

    eval_qa = []
    best_generated_answer = []
    # Create a prompt for generate GPT-3 grade label
    prompts = list(eval_dataframe['prompt'])
    completions = list(eval_dataframe['completion'])
    for question, ans in tqdm(zip(prompts, completions),
                              desc='Running OpenAssistant',
                              total=len(prompts),
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
      temp_q_dict = {}
      temp_new_answer_dict = {}
      temp_q_dict['question'] = question
      temp_q_dict['answer'] = ans

      # generate answer using OpenAssistant
      generated_answer = self.open_assistant(prompt_template, question)
      # print("\nGenerated answer:")
      # print(generated_answer)
      # previous T5 question answer pipeline
      # generated_answers, _ = self.ta_pipeline.question_answer(question, "")
      # temp_new_answer_dict['text'] = generated_answers["Answer"].head(1).values

      temp_new_answer_dict['text'] = generated_answer
      eval_qa.append(temp_q_dict)
      best_generated_answer.append(temp_new_answer_dict)

    # RUN LangChain GPT-3 evaluation
    # TODO: add some context in template of GPT-3
    _PROMPT_TEMPLATE = """You are an expert teaching assistant specialized in evaluating two students' answers in response to the instructor's question.
                          You are referring to the following question:
                          {query}
                          Here is the answer from student 1:
                          {answer}
                          Here is the answer from student 2:
                          {result}
                          Please consider the relevance, accuracy, correctness, and fluency of their responses. The answer which is concise but includes specific detailed examples is preferable in evaluation. 
                          You need to avoid any potential bias in your evaluation and ensure that the order in which the responses were presented does not affect your judgment.
                          Based on student 1, please output a label for student 2 as "Better", "Worse" or "Same".
                          """
    gpt3_eval_prompt = PromptTemplate(input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE)
    eval_model = OpenAI(temperature=0)
    evalchain = QAEvalChain.from_llm(llm=eval_model, prompt=gpt3_eval_prompt)
    # Grade the new model generated answer compared to the original one
    grader = evalchain.evaluate(eval_qa, best_generated_answer, question_key="question", answer_key="answer", prediction_key="text")

    # Add the new evaluation results to a new evaluation set (w/ two answers version)
    # and the original evaluation set (cover the worse answers)
    new_eval_set = []
    num_better = 0
    num_worse = 0
    num_same = 0
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

      # just for quick debugging.
      if grade_label == 'Better':
        num_better += 1
      elif grade_label == 'Worse':
        num_worse += 1
      elif grade_label == 'Same':
        num_same += 1
      print(f"\t\tFraction of answers that are 'better' {eval_name}: {num_better / (i+1)}")

    print(f"\n\n🆗 Fraction of answers that are 'same' {eval_name}: {num_same / NUM_OF_DATAPOINTS_TO_EVALUATE}\n\n")
    print(f"\n\n✅ Fraction of answers that are 'better' {eval_name}: {num_better / NUM_OF_DATAPOINTS_TO_EVALUATE}\n\n")

    # Write the new evaluation data to the JSON file
    file_name = "./eval_results/" + make_workflow_id(eval_name) + ".json"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # Write the new evaluation data (w/ two compared answers verision) to the JSON file
    # The format of the JSON file includes: question, original answer, chatbot generated answer, GPT-3 evaluation label
    with open(file_name, 'w', encoding='utf-8') as f:
      json.dump(new_eval_set, f, ensure_ascii=False, indent=4)

  def rouge_n_bleu_score(self, eval_dataset):
    """
    DEPRICATED -- less useful than the GPT-3 evaluation.
    
    Args:user_question (str): questions from the human filtered eval set

    Returns: the evaluation score for each user's question compared to the human labeled answers
    
    overall_rouge_score: the average RougeL f1 score for all the questions in eval set
    overall_bleu_score: the average Bleu1 score for all the questions in eval set
    """
    # eval_dataset = json.load(open(eval_set_path, 'r'))
    bleu_metric = evaluate.load('bleu')
    rouge = Rouge()
    rouge_score_list, bleu_score_list = [], []

    eval_dataframe = pd.DataFrame()
    eval_dataframe['prompt'] = eval_dataset['prompt'][:NUM_OF_DATAPOINTS_TO_EVALUATE]
    eval_dataframe['completion'] = eval_dataset['completion'][:NUM_OF_DATAPOINTS_TO_EVALUATE]

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


def make_workflow_id(name: str) -> str:
  '''
  🎯 Best practice to ensure unique Workflow names.
  '''
  from datetime import datetime

  import pytz

  # Timezones: US/{Pacific, Mountain, Central, Eastern}
  # All timezones `pytz.all_timezones`. Always use caution with timezones.
  curr_time = datetime.now(pytz.timezone('US/Central'))
  return f"{name}-{str(curr_time.strftime('%h_%d,%Y@%H:%M'))}"


def main():
  eval_dataset = datasets.load_dataset(
      "kastan/rlhf-qa-conditional-generation-v2",
      split="train+valid",
  )
  global NUM_OF_DATAPOINTS_TO_EVALUATE
  NUM_OF_DATAPOINTS_TO_EVALUATE = min(NUM_OF_DATAPOINTS_TO_EVALUATE, len(eval_dataset['prompt']))
  evaluator = Evaluator()
  evaluator.main_eval_loop(eval_dataset)


if __name__ == '__main__':
  main()

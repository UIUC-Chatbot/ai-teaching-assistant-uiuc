import os
import sys

ROOT_DIR = os.path.abspath("../retreival-generation-system/trt_accelerate/HuggingFace/")
sys.path.append(ROOT_DIR)
sys.path.append("../human_data_review")
sys.path.append("../retreival-generation-system")
sys.path.append("../retreival-generation-system/trt_accelerate")
import argparse
import json
import pprint
import random
import time
from datetime import datetime
from typing import Dict, List

import gradio as gr
import numpy as np
import pandas as pd
import torch
from langchain.chains import LLMChain
from langchain.evaluation.qa import QAEvalChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from PIL import Image

import main
import wandb
from gpu_memory_utils import (get_device_with_most_free_memory, get_gpu_ids_with_sufficient_memory)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ensure previous sessions are closed from our public port 8888
gr.close_all()

# from .autonotebook import tqdm as notebook_tqdm

# Todo: integrate CLIP.
# Todo: log images.
# wandb.log(
#     {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]},
#     step=0)

<<<<<<< HEAD
NUM_ANSWERS_GENERATED = 2
=======
NUM_ANSWERS_GENERATED = 2 
>>>>>>> 16a7f975601810a373b698d0b771d3926a34a395
NUM_ANSWERS_TO_SHOW_USER = 3
NUM_IMAGES_TO_SHOW_USER = 4  # 4 is good for gradio image layout


def main_arg_parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_weight', type=str, default=None)
  # parser.add_argument('--device', type=str, default='cuda:0') # should ALWAYS be dynamically selected...
  parser.add_argument('--wandb_entity', type=str, default='uiuc-ta-chatbot-team')
  parser.add_argument('--wandb_project', type=str, default="First_TA_Chatbot")
  # parser.add_argument('--trt_path',type = str, default= None)
  args = parser.parse_args()
  return args


import torch.autograd.profiler as profiler


class TA_Gradio():

<<<<<<< HEAD
  def __init__(self, args):
    # dynamically select device based on available GPU memory
    self.device = torch.device(f'cuda:{get_device_with_most_free_memory()}')
    opt_device_list = get_gpu_ids_with_sufficient_memory(24)  # at least 24GB of memory
=======
    def __init__(self, args):
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        self.device = torch.device(args.device)
        self.ta = main.TA_Pipeline(
            device=self.device,
            opt_weight_path=args.model_weight,
            ct2_path = "../data/models/opt_acc/opt_1.3b_fp16",
            is_server = True,
            device_index = [0,1],
            n_stream = 2
            )  
        # accelerate OPT model (optimized model with multiple instances, parallel execution): 
        # ct2_path = "../data/models/opt_acc/opt_1.3b_fp16",
        # is_server = True,
        # device_index = [0,1],
        # n_stream = 3
>>>>>>> 16a7f975601810a373b698d0b771d3926a34a395

    self.ta = main.TA_Pipeline(opt_weight_path=args.model_weight,
                               ct2_path="../data/models/opt_acc/opt_1.3b_fp16",
                               is_server=True,
                               device_index_list=opt_device_list)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

  def run_clip(self, user_question: str, num_images_returned: int = 4):
    return self.ta.clip(user_question, num_images_returned)

  def model_evaluation(self,
                       eval_set_path: str = '/home/zhiweny2/chatbotai/jerome/human_data_review/gpt-3_semantic_search/1_top_quality.json'):
    """
    Args: evaluation set path: dataset path for GPT-3 evaluation

    Returns: output TWO result files
    1. Generate a .json file which contains {question, original answer, new generated answer, GPT-3 grade}
    2. Generate a new .json eval set, which contains {question, new better answer}
    
    Change each file path you would like to evaluate or generate
    """

    self.eval_set_path = eval_set_path
    eval_set = json.load(open(self.eval_set_path, 'r'))
    eval_qa = []
    best_generated_answer = []
    # Create a prompt for generate GPT-3 grade label
    _PROMPT_TEMPLATE = """You are an expert professor specialized in evaluating students' new answers comparing to their previous answers given the same questions.
                            You are referring the following question:
                            {query}
                            Here is the previous answer:
                            {answer}
                            You are evaluating the following new answer:
                            {result}
                            Do you think the new answer is better than the previous answer? Label as "Better" or "Worse".
                            """
<<<<<<< HEAD
    PROMPT = PromptTemplate(input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE)
=======
        PROMPT = PromptTemplate(input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE)
        
        # Process eval set to match LangChain requirements
        for dataset in [eval_set]:
            for row in dataset:
                temp_q_dict = {}
                temp_new_answer_dict = {}
                temp_question = row['GPT-3-Generations']['question']
                temp_q_dict['question'] = temp_question
                temp_q_dict['answer'] = row['GPT-3-Generations']['answer']
                generated_answers, _ = self.question_answer(temp_question, "")
                temp_new_answer_dict['text'] = generated_answers["Answer"].head(1).values
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
        updated_eval_set = []
        for i, row in enumerate(eval_set):
            new_generated_answer = best_generated_answer[i]['text']
            grade_label = grader[i]['text'].replace('\n', '')
            # Create a new evaluation set with the new generated answer        
            row['Chatbot-Generated-Answer'] = new_generated_answer
            row['GPT-3-Evaluation'] = grade_label
            new_eval_set.append(row)
            # Rewrite the original evluation set with the new generated 'Better' answer
            if 'Better' in grade_label:
                row['GPT-3-Generations']['answer'] = new_generated_answer
                updated_eval_set.append(row)
            else:
                updated_eval_set.append(row) 
                
        # Write the new evaluation data to the JSON file
        # Get the current date and time
        now = datetime.now()
        # Format the date and time as a string
        timestamp = now.strftime("%Y-%m-%d_%H-%M")
        # Create a file name with the date and time as a suffix
        file_name = "/home/zhiweny2/chatbotai/jerome/human_data_review/" + "gpt3_graded_set_" + timestamp + ".json"
        # Write the new evaluation data (w/ two compared answers verision) to the JSON file
        # The format of the JSON file includes: question, original answer, chatbot generated answer, GPT-3 evaluation label
        # Change the path you want to save this file for human comparision only
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(new_eval_set, f, ensure_ascii=False, indent=4) 
        # Write the updated evaluation data to the JSON file 
        # Change the path you want to save this updated eval set for further evaluation
        with open('/home/zhiweny2/chatbotai/jerome/human_data_review/new_eval_set.json', 'w', encoding='utf-8') as f:
            json.dump(updated_eval_set, f, ensure_ascii=False, indent=4) 
    
    def question_answer(self, question: str, user_defined_context: str = '', use_gpt3: bool = False, image=None):
        """
        This is the function called with the user clicks the main "Search 🔍" button.
        You can call this from anywhere to run our main program.
        
        question: user-supplied question
        [OPTIONAL] user_defined_context: user-supplied context to make the answer more specific. Usually it's empty, so we AI retrieve a context.
        [OPTIONAL] use_gpt3: Run GPT-3 answer-generation if True, default is False. The True/False value of the checkbox in the UI to "Use GPT3 (paid)". 
        [OPTIONAL] image: User-supplied image, for reverse image search.
        """
        start_time = time.monotonic()
        # we generate many answers, then filter it down to the best scoring ones (w/ msmarco).
        
        USER_QUESTION = str(question)
        print("-----------------------------\nINPUT USER QUESTION:", USER_QUESTION, '\n-----------------------------')
>>>>>>> 16a7f975601810a373b698d0b771d3926a34a395

    # Process eval set to match LangChain requirements
    for dataset in [eval_set]:
      for row in dataset:
        temp_q_dict = {}
        temp_new_answer_dict = {}
        temp_question = row['GPT-3-Generations']['question']
        temp_q_dict['question'] = temp_question
        temp_q_dict['answer'] = row['GPT-3-Generations']['answer']
        generated_answers, _ = self.question_answer(temp_question, "")
        temp_new_answer_dict['text'] = generated_answers["Answer"].head(1).values
        eval_qa.append(temp_q_dict)
        best_generated_answer.append(temp_new_answer_dict)

    # Load LangChain Evaluation pipeline
    eval_model = OpenAI(temperature=0)
    evalchain = QAEvalChain.from_llm(llm=eval_model, prompt=PROMPT)
    # Grade the new model generated answer compared to the original one
    grader = evalchain.evaluate(eval_qa, best_generated_answer, question_key="question", answer_key="answer", prediction_key="text")

<<<<<<< HEAD
    # Add the new evaluation results to a new evaluation set (w/ two answers version)
    # and the original evaluation set (cover the worse answers)
    new_eval_set = []
    updated_eval_set = []
    for i, row in enumerate(eval_set):
      new_generated_answer = best_generated_answer[i]['text']
      grade_label = grader[i]['text'].replace('\n', '')
      # Create a new evaluation set with the new generated answer
      row['Chatbot-Generated-Answer'] = new_generated_answer
      row['GPT-3-Evaluation'] = grade_label
      new_eval_set.append(row)
      # Rewrite the original evluation set with the new generated 'Better' answer
      if 'Better' in grade_label:
        row['GPT-3-Generations']['answer'] = new_generated_answer
        updated_eval_set.append(row)
      else:
        updated_eval_set.append(row)

    # Write the new evaluation data to the JSON file
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as a string
    timestamp = now.strftime("%Y-%m-%d_%H-%M")
    # Create a file name with the date and time as a suffix
    file_name = "/home/zhiweny2/chatbotai/jerome/human_data_review/" + "gpt3_graded_set_" + timestamp + ".json"
    # Write the new evaluation data (w/ two compared answers verision) to the JSON file
    # The format of the JSON file includes: question, original answer, chatbot generated answer, GPT-3 evaluation label
    # Change the path you want to save this file for human comparision only
    with open(file_name, 'w', encoding='utf-8') as f:
      json.dump(new_eval_set, f, ensure_ascii=False, indent=4)
    # Write the updated evaluation data to the JSON file
    # Change the path you want to save this updated eval set for further evaluation
    with open('/home/zhiweny2/chatbotai/jerome/human_data_review/new_eval_set.json', 'w', encoding='utf-8') as f:
      json.dump(updated_eval_set, f, ensure_ascii=False, indent=4)

  # def question_answer(self, question: str, user_defined_context: str = '', use_gpt3: bool = False, image=None):
  #   """
  #   This is the function called with the user clicks the main "Search 🔍" button.
  #   You can call this from anywhere to run our main program.

  #   question: user-supplied question
  #   [OPTIONAL] user_defined_context: user-supplied context to make the answer more specific. Usually it's empty, so we AI retrieve a context.
  #   [OPTIONAL] use_gpt3: Run GPT-3 answer-generation if True, default is False. The True/False value of the checkbox in the UI to "Use GPT3 (paid)".
  #   [OPTIONAL] image: User-supplied image, for reverse image search.
  #   """
  #   start_time = time.monotonic()
  #   # we generate many answers, then filter it down to the best scoring ones (w/ msmarco).

  #   USER_QUESTION = str(question)
  #   print("-----------------------------\nINPUT USER QUESTION:", USER_QUESTION, '\n-----------------------------')

  #   # check if user supplied their own context.
  #   if len(user_defined_context) == 0:
  #     # retrieve contexts
  #     start_time_pinecone = time.monotonic()
  #     top_context_documents = self.ta.retrieve_contexts_from_pinecone(user_question=USER_QUESTION, topk=NUM_ANSWERS_GENERATED)
  #     top_context_metadata = [
  #         f"Source: page {int(doc.metadata['page_number'])} in {doc.metadata['textbook_name']}" for doc in top_context_documents
  #     ]
  #     top_context_list = [doc.page_content for doc in top_context_documents]
  #     print(f"⏰ Runtime for Pinecone: {(time.monotonic() - start_time_pinecone):.2f} seconds")
  #     # print(doc.metadata['page_number'], doc.metadata['textbook_name'])

  #     # TODO: add OPT back in when Wentao is ready.
  #     # Run opt answer generation
  #     # generated_answers_list = self.ta.OPT(USER_QUESTION,
  #     #                                      top_context_list,
  #     #                                      NUM_ANSWERS_GENERATED,
  #     #                                      print_answers_to_stdout=False)

  #     # T5 generations
  #     generated_answers_list = []
  #     generated_answers_list.extend(
  #         self.ta.run_t5_completion(USER_QUESTION,
  #                                   top_context_list,
  #                                   num_answers_generated=NUM_ANSWERS_GENERATED,
  #                                   print_answers_to_stdout=True))

  #     print("GENERATED ANS LIST: ", generated_answers_list)
  #     yield generated_answers_list

  #   else:
  #     # TODO: add OPT back in when Wentao is ready.
  #     # opt: passage + question --> answer
  #     # generated_answers_list = self.ta.OPT_one_question_multiple_answers(
  #     #     USER_QUESTION,
  #     #     user_defined_context,
  #     #     num_answers_generated=NUM_ANSWERS_GENERATED,
  #     #     print_answers_to_stdout=False)

  #     # T5 generations
  #     generated_answers_list = []
  #     generated_answers_list.extend(
  #         self.ta.run_t5_completion(USER_QUESTION,
  #                                   user_defined_context,
  #                                   num_answers_generated=NUM_ANSWERS_GENERATED,
  #                                   print_answers_to_stdout=True))

  #     yield generated_answers_list
  # show (the same) user-supplied context for next to each generated answer.
  # top_context_list = [user_defined_context] * NUM_ANSWERS_GENERATED

  #print("GENERATED ANSWER: ", generated_answers_list)
  # rank potential answers
  # todo: rank both!!
  #final_scores = self.ta.re_ranking_ms_marco(generated_answers_list[NUM_ANSWERS_GENERATED:], USER_QUESTION)

  # return a pd datafarme, to display a gr.dataframe
  # results = {
  #     'Answer': generated_answers_list[NUM_ANSWERS_GENERATED:],
  #     # append page number and textbook name to each context
  #     'Context': [f"{text}. {meta}" for text, meta in zip(top_context_list, top_context_metadata)],
  #     # 'Context': top_context_list,
  #     'Score': final_scores,
  # }
  # print(len(generated_answers_list))
  # print(len(top_context_list))
  # print(len(final_scores))

  #print("RESULTS: ", results)
=======
            # TODO: add OPT back in when Wentao is ready.
            # Run opt answer generation
            # generated_answers_list = self.ta.OPT(USER_QUESTION,
            #                                      top_context_list,
            #                                      NUM_ANSWERS_GENERATED,
            #                                      print_answers_to_stdout=False)

            # T5 generations
            generated_answers_list = []
            generated_answers_list.extend(self.ta.run_t5_completion(USER_QUESTION,
                                                               top_context_list,
                                                               num_answers_generated=NUM_ANSWERS_GENERATED,
                                                               print_answers_to_stdout=True))
            
            print("GENERATED ANS LIST: ", generated_answers_list)
            yield generated_answers_list

            

        else:
            # TODO: add OPT back in when Wentao is ready.
            # opt: passage + question --> answer
            generated_answers_list = self.ta.OPT_one_question_multiple_answers(
                USER_QUESTION,
                user_defined_context,
                num_answers_generated=NUM_ANSWERS_GENERATED,
                print_answers_to_stdout=False)

            # T5 generations
            generated_answers_list.extend(self.ta.run_t5_completion(USER_QUESTION,
                                                               user_defined_context,
                                                               num_answers_generated=NUM_ANSWERS_GENERATED,
                                                               print_answers_to_stdout=True))
            
                        

            # show (the same) user-supplied context for next to each generated answer.
            top_context_list = [user_defined_context] * NUM_ANSWERS_GENERATED
        
        
        #print("GENERATED ANSWER: ", generated_answers_list)
        # rank potential answers
        # todo: rank both!!
        #final_scores = self.ta.re_ranking_ms_marco(generated_answers_list[NUM_ANSWERS_GENERATED:], USER_QUESTION)

        # return a pd datafarme, to display a gr.dataframe
        # results = {
        #     'Answer': generated_answers_list[NUM_ANSWERS_GENERATED:],
        #     # append page number and textbook name to each context
        #     'Context': [f"{text}. {meta}" for text, meta in zip(top_context_list, top_context_metadata)],
        #     # 'Context': top_context_list,
        #     'Score': final_scores,
        # }
        # print(len(generated_answers_list))
        # print(len(top_context_list))
        # print(len(final_scores))

        #print("RESULTS: ", results)

        # sort results by MSMarco ranking
        # generated_results_df = pd.DataFrame(results).sort_values(by=['Score'],
        #                                                          ascending=False).head(NUM_ANSWERS_TO_SHOW_USER)

        # GPT3 for comparison to SOTA. Append to df to ensure it's ALWAYS displayed, regardless of msmarco score.
        # if use_gpt3:
        #     generated_results_df = self.add_gpt3_response(generated_results_df, USER_QUESTION, top_context_list)

        # todo: include gpt3 results in logs. generated_results_df to wandb.
        # append data to wandb
        # self.log_results_to_wandb(USER_QUESTION, generated_answers_list, final_scores, top_context_list,
        #                           user_defined_context,
        #                           time.monotonic() - start_time)

        #print("DF PRINT", generated_results_df)


        # Flag for if we want to use CLIP or not.
        # use_clip = False  
        # if use_clip:
        #     return generated_results_df, self.run_clip(question, NUM_IMAGES_TO_SHOW_USER)
        # else:
        #     # without running clip
        #     return generated_results_df.Answer[0], None

    # def log_results_to_wandb(self, user_question, generated_answers_list, final_scores, top_context_list,
    #                          user_defined_context, runtime) -> None:
    #     wandb.log({'runtime (seconds)': runtime})

    #     results_table = wandb.Table(columns=[
    #         "question", "user_supplied_context", "generated_answers", "retrieved_contexts", "scores",
    #         "runtime (seconds)"
    #     ])
    #     for ans, score, retrieved_context in zip(generated_answers_list, final_scores, top_context_list):
    #         one_row_of_data = [user_question, user_defined_context, ans, retrieved_context, score, runtime]
    #         results_table.add_data(*one_row_of_data)

    #     # log a new table for each time our app is used. Can't figure out how to append to them easily.
    #     wandb.log({make_inference_id('Inference_made'): results_table})

    # def add_gpt3_response(self, results_df: pd.DataFrame, user_question, top_context_list: List[str]) -> pd.DataFrame:
    #     """
    #     GPT3 for comparison to SOTA.
    #     This answer is ALWAYS shown to the user, no matter the score. It is not subject to score filtering like the other generations are.
    #     """
    #     generated_answer = "GPT-3 response:\n" + self.ta.gpt3_completion(user_question, top_context_list[0])

    #     score = self.ta.re_ranking_ms_marco([generated_answer], user_question)

    #     gpt3_result = {
    #         'Answer': [generated_answer],
    #         'Context': [top_context_list[0]],
    #         'Score': score,  # score is already a list
    #     }
    #     df_to_append = pd.DataFrame(gpt3_result)
    #     return pd.concat([df_to_append, results_df], ignore_index=True)
>>>>>>> 16a7f975601810a373b698d0b771d3926a34a395

  # sort results by MSMarco ranking
  # generated_results_df = pd.DataFrame(results).sort_values(by=['Score'],
  #                                                          ascending=False).head(NUM_ANSWERS_TO_SHOW_USER)

<<<<<<< HEAD
  # GPT3 for comparison to SOTA. Append to df to ensure it's ALWAYS displayed, regardless of msmarco score.
  # if use_gpt3:
  #     generated_results_df = self.add_gpt3_response(generated_results_df, USER_QUESTION, top_context_list)
=======
        user_utter, topic, topic_history = self.ta.et_main(message)
        print("Topic:", topic)
        psg = self.ta.retrieve(user_utter, 1)
        out_ans = self.ta.OPT(user_utter, psg, 1, False)[0]
        self.ta.et_add_ans(out_ans)
        final_out = "[RESPONSE]:" + out_ans
        history.append((message, final_out))
        return history
    
    def load_text_answer(self, question, context):
        self.generated_answers_list = []
        self.retrieved_context_list = []
        for i, ans in enumerate(self.ta.yield_text_answer(question, context)):
            print("IN LOAD TEXT ANSWER")
            i = 2*i
            ans_list = [gr.update() for j in range(7)]
            ans_list[i] = gr.update(value=ans[0])
            ans_list[i+1] = gr.update(value=ans[1])

            print(ans_list)
            self.generated_answers_list.append(ans[0])
            self.retrieved_context_list.append(ans[1])
            yield ans_list

        # call ranking function here
        final_scores = self.ta.re_ranking_ms_marco(self.generated_answers_list, question)
        print(final_scores)

        results = {
            'Answer': self.generated_answers_list,
            # append page number and textbook name to each context
            #'Context': [f"{text}. {meta}" for text, meta in zip(top_context_list, top_context_metadata)],
            # 'Context': top_context_list,
            'Score': final_scores,
        }

        generated_results_df = pd.DataFrame(results).sort_values(by=['Score'],
                                                                 ascending=False).head(NUM_ANSWERS_TO_SHOW_USER)

        generated_results_df = generated_results_df.reset_index()
        print("DF: ", generated_results_df)
        new_list = [gr.update() for j in range(7)]
        new_list[-1] = gr.update(value = str(generated_results_df['Answer'][0]))
        print(new_list)
        yield new_list 
    
>>>>>>> 16a7f975601810a373b698d0b771d3926a34a395

  # todo: include gpt3 results in logs. generated_results_df to wandb.
  # append data to wandb
  # self.log_results_to_wandb(USER_QUESTION, generated_answers_list, final_scores, top_context_list,
  #                           user_defined_context,
  #                           time.monotonic() - start_time)

  #print("DF PRINT", generated_results_df)

  # Flag for if we want to use CLIP or not.
  # use_clip = False
  # if use_clip:
  #     return generated_results_df, self.run_clip(question, NUM_IMAGES_TO_SHOW_USER)
  # else:
  #     # without running clip
  #     return generated_results_df.Answer[0], None

  # def log_results_to_wandb(self, user_question, generated_answers_list, final_scores, top_context_list,
  #                          user_defined_context, runtime) -> None:
  #     wandb.log({'runtime (seconds)': runtime})

  #     results_table = wandb.Table(columns=[
  #         "question", "user_supplied_context", "generated_answers", "retrieved_contexts", "scores",
  #         "runtime (seconds)"
  #     ])
  #     for ans, score, retrieved_context in zip(generated_answers_list, final_scores, top_context_list):
  #         one_row_of_data = [user_question, user_defined_context, ans, retrieved_context, score, runtime]
  #         results_table.add_data(*one_row_of_data)

  #     # log a new table for each time our app is used. Can't figure out how to append to them easily.
  #     wandb.log({make_inference_id('Inference_made'): results_table})

  # def add_gpt3_response(self, results_df: pd.DataFrame, user_question, top_context_list: List[str]) -> pd.DataFrame:
  #     """
  #     GPT3 for comparison to SOTA.
  #     This answer is ALWAYS shown to the user, no matter the score. It is not subject to score filtering like the other generations are.
  #     """
  #     generated_answer = "GPT-3 response:\n" + self.ta.gpt3_completion(user_question, top_context_list[0])

  #     score = self.ta.re_ranking_ms_marco([generated_answer], user_question)

  #     gpt3_result = {
  #         'Answer': [generated_answer],
  #         'Context': [top_context_list[0]],
  #         'Score': score,  # score is already a list
  #     }
  #     df_to_append = pd.DataFrame(gpt3_result)
  #     return pd.concat([df_to_append, results_df], ignore_index=True)

  def chat(self, message, history):
    history = history or []

    user_utter, topic, topic_history = self.ta.et_main(message)
    print("Topic:", topic)
    psg = self.ta.retrieve(user_utter, 1)
    out_ans = self.ta.OPT(user_utter, psg, 1, False)[0]
    self.ta.et_add_ans(out_ans)
    final_out = "[RESPONSE]:" + out_ans
    history.append((message, final_out))
    return history

  def load_text_answer(self, question, context):
    '''
    This function is called when the user clicks the "Generate Answer" button.
    It collects responses and updates the gradio interface iteratively as we get new responses. 
    At the end, it shows a 'main answer' after all answers are generated AND ranked.
    '''
    self.generated_answers_list = []
    self.retrieved_context_list = []
    for i, ans in enumerate(self.ta.yield_text_answer(question, context)):
      # print("IN LOAD TEXT ANSWER")
      i = 2 * i
      ans_list = [gr.update() for j in range(7)]
      ans_list[i] = gr.update(value=ans[0])
      ans_list[i + 1] = gr.update(value=ans[1])

      # print(ans_list)
      self.generated_answers_list.append(ans[0])
      self.retrieved_context_list.append(ans[1])
      yield ans_list

    # call ranking function here
    final_scores = self.ta.re_ranking_ms_marco(self.generated_answers_list, question)
    # print(final_scores)

    results = {
        'Answer': self.generated_answers_list,
        # append page number and textbook name to each context
        #'Context': [f"{text}. {meta}" for text, meta in zip(top_context_list, top_context_metadata)],
        # 'Context': top_context_list,
        'Score': final_scores,
    }

    generated_results_df = pd.DataFrame(results).sort_values(by=['Score'], ascending=False).head(NUM_ANSWERS_TO_SHOW_USER)

    generated_results_df = generated_results_df.reset_index()
    # print("DF: ", generated_results_df)
    new_list = [gr.update() for j in range(7)]
    new_list[-1] = gr.update(value=str(generated_results_df['Answer'][0]))
    # print(new_list)
    yield new_list

  def main(self,):
    with gr.Blocks() as input_blocks:
      # title and description or use gr.HTML(...)
      gr.Markdown("""# Ask an Electrical Engineering Question
                        #### Our system will answer your question directly, and give links to all your course materials.
                        """)
      flagging_dir = 'user-flagged-to-review',
      ''' Main user input section '''
      with gr.Row():
        with gr.Column(scale=2.6):
          search_question = gr.Textbox(label="Search\n", placeholder="Ask me anything...")
          context = gr.Textbox(label="(Optional) give a relevant textbook paragraph for specific questions",
                               placeholder="(Optional) we'll use the paragraph to generate an answer to your question.")
          # gr.Markdown("""Try searching for:""")
          use_gpt3_checkbox = gr.Checkbox(label="Include GPT-3 (paid)?")
          examples = gr.Examples(
              examples=[
                  ["What is a Finite State Machine?"],
                  ["How do you design a functional a Two-Bit Gray Code Counter?"],
              ],
              inputs=[search_question, context],  # todo: fix img part
              outputs=[],
          )
        # reverse image search
        image = gr.Image(type="pil", label="[NOT IMPLEMENTED YET] -- Reverse Image Search (optional)", shape=(224, 224))
      ''' Button and on-click function '''
      with gr.Row():
        # create a button with an orange background
        # run = gr.Button("Search 🔍", style='')
        run = gr.Button(
            "Search  🔍",
            variant='primary',
        )
        # run_reverse_img_search = gr.Button("Image search", variant='secondary',)
      # with gr.Row():
      # event = run.click(fn=self.question_answer,
      #                   inputs=[search_question, context, use_gpt3_checkbox, image],
      #                   outputs=[
      #                       gr.Dataframe(
      #                           headers=["Answer", "Score", "Context", "Metadata"],
      #                           wrap=True,
      #                       ),
      #                       gr.Gallery(label="Lecture images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")
      #                   ],
      #                   scroll_to_output=True)
      ''' RESULTS SECTION for text answers '''
      with gr.Row():
        gr.Markdown("""## Results""")
        best_answer = gr.Textbox(label="Best Answer", wrap=True)  # scroll_to_output=True

<<<<<<< HEAD
      with gr.Row():
        with gr.Column():
          generated_answer1 = gr.Textbox(label="Answer 1", wrap=True)
          context1 = gr.Textbox(label="Context 1", wrap=True)

          feedback_radio1 = gr.Radio(['Like', 'Dislike'], label="Feedback")
          custom_ans1 = gr.Textbox(label="Custom Answer", input="text")
        with gr.Column():
          generated_answer2 = gr.Textbox(label="Answer 2", wrap=True)
          context2 = gr.Textbox(label="Context 2", wrap=True)
=======
            with gr.Row():
                best_answer = gr.Textbox(label="Best Answer", wrap=True)

            with gr.Row():
                with gr.Column():
                    generated_answer1 = gr.Textbox(label="Answer 1", wrap=True)
                    context1 = gr.Textbox(label="Context 1", wrap=True)
                    
                    feedback_radio1 = gr.Radio(['Like', 'Dislike'], label="Feedback")
                    custom_ans1 = gr.Textbox(label="Custom Answer", input="text")
                with gr.Column():
                    generated_answer2 = gr.Textbox(label="Answer 2", wrap=True)
                    context2 = gr.Textbox(label="Context 2", wrap=True)
                    
                    feedback_radio2 = gr.Radio(['Like', 'Dislike'], label="Feedback")
                    custom_ans2 = gr.Textbox(label="Custom Answer", input="text")
                with gr.Column():
                    generated_answer3 = gr.Textbox(label="Answer 3", wrap=True)
                    context3 = gr.Textbox(label="Context 3", wrap=True)

                    feedback_radio3 = gr.Radio(['Like', 'Dislike'], label="Feedback")
                    custom_ans3 = gr.Textbox(label="Custom Answer", input="text")
            
            with gr.Row():
                feedback_btn = gr.Button(value="submit")
                feedback_btn.click(save_feedback, inputs=[search_question, generated_answer1, context1, feedback_radio1,
                                                          custom_ans1, generated_answer2, context2, feedback_radio2, custom_ans2,
                                                          generated_answer3, context3, feedback_radio3, custom_ans3])

            with gr.Row():
                lec_gallery = gr.Gallery(label="Lecture images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")

                

            #run.click(fn=self.answer_loading, inputs=[search_question, context, use_gpt3_checkbox, image])

            # event = run.click(fn=self.question_answer,
            #                   inputs=[search_question, context, use_gpt3_checkbox, image],
            #                   outputs=[generated_answer, lec_gallery],
            #                   scroll_to_output=True)

            run.click(fn=self.load_text_answer, inputs=[search_question, context],
                       outputs=[generated_answer1, context1, generated_answer2, context2, 
                                generated_answer3, context3, best_answer])
    
            with gr.Row():
                feedback_radio = gr.Radio(['Like', 'Dislike'], label="Feedback")
                custom_ans = gr.Textbox(label="Custom Answer", input="text")
                feedback_btn = gr.Button(value="submit")
                feedback_btn.click(save_feedback, inputs=[feedback_radio, custom_ans, search_question, generated_answer1])
            with gr.Row():
                txt = gr.Textbox(label="chat", lines=2)
                chatbot = gr.Chatbot().style(color_map=("green", "pink"))
            with gr.Row():
                chat = gr.Button("Chat", variant='primary')
>>>>>>> 16a7f975601810a373b698d0b771d3926a34a395

          feedback_radio2 = gr.Radio(['Like', 'Dislike'], label="Feedback")
          custom_ans2 = gr.Textbox(label="Custom Answer", input="text")
        with gr.Column():
          generated_answer3 = gr.Textbox(label="Answer 3", wrap=True)
          context3 = gr.Textbox(label="Context 3", wrap=True)

          feedback_radio3 = gr.Radio(['Like', 'Dislike'], label="Feedback")
          custom_ans3 = gr.Textbox(label="Custom Answer", input="text")

      with gr.Row():
        feedback_btn = gr.Button(value="submit")
        feedback_btn.click(save_feedback,
                           inputs=[
                               search_question, generated_answer1, context1, feedback_radio1, custom_ans1, generated_answer2, context2,
                               feedback_radio2, custom_ans2, generated_answer3, context3, feedback_radio3, custom_ans3
                           ])

      with gr.Row():
        lec_gallery = gr.Gallery(label="Lecture images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")

        #run.click(fn=self.answer_loading, inputs=[search_question, context, use_gpt3_checkbox, image])

        # event = run.click(fn=self.question_answer,
        #                   inputs=[search_question, context, use_gpt3_checkbox, image],
        #                   outputs=[generated_answer, lec_gallery],
        #                   scroll_to_output=True)

        run.click(fn=self.load_text_answer,
                  inputs=[search_question, context],
                  outputs=[generated_answer1, context1, generated_answer2, context2, generated_answer3, context3, best_answer])

        # with gr.Row():
        #   feedback_radio = gr.Radio(['Like', 'Dislike'], label="Feedback")
        #   custom_ans = gr.Textbox(label="Custom Answer", input="text")
        #   feedback_btn = gr.Button(value="submit")
        #   feedback_btn.click(save_feedback, inputs=[feedback_radio, custom_ans, search_question, generated_answer1])
        # with gr.Row():
        #   txt = gr.Textbox(label="chat", lines=2)
        #   chatbot = gr.Chatbot().style(color_map=("green", "pink"))
        # with gr.Row():
        #   chat = gr.Button("Chat", variant='primary')

    # demo = gr.Interface(
    #     self.chat,
    #     ["text", "state"],
    #     [chatbot, "state"],
    #     allow_flagging="never",
    # )
    input_blocks.queue(concurrency_count=2)  # limit concurrency
    # input_blocks.launch(share=True)  #, server_port=8888)
    input_blocks.launch(share=True)  #, server_name='0.0.0.0', server_port=8889)
    input_blocks.integrate(wandb=wandb)


def save_feedback(query, answer1, context1, likes1, custom_answer1, answer2, context2, likes2, custom_answer2, answer3, context3, likes3,
                  custom_answer3):
  new_data = {
      'gradio_feedback': [{
          'question': query,
          'generated_answer_1': answer1,
          'context_1': context1,
          'feedback_1': likes1,
          'custom_answer_1': custom_answer1,
          'generated_answer_2': answer2,
          'context_2': context2,
          'feedback_2': likes2,
          'custom_answer_2': custom_answer2,
          'generated_answer_3': answer3,
          'context_3': context3,
          'feedback_3': likes3,
          'custom_answer_3': custom_answer3
      }]
  }
  # save to csv --> get question and answers here.
  filepath = "feedback.json"
  if os.path.exists(filepath):
    with open("feedback.json", "r+") as f:
      file_data = json.load(f)
      file_data['gradio_feedback'].append(new_data['gradio_feedback'][0])
      f.seek(0)
      json.dump(file_data, f, indent=4)
  else:
    with open("feedback.json", "w") as f:
      json.dump(new_data, f)



def save_feedback(query, answer1, context1, likes1, custom_answer1, answer2, context2, likes2, custom_answer2, answer3, context3, likes3, custom_answer3):
    new_data = {'gradio_feedback': [{
        'question': query,
        'generated_answer_1': answer1,
        'context_1': context1,
        'feedback_1': likes1,
        'custom_answer_1': custom_answer1,
        'generated_answer_2': answer2,
        'context_2': context2,
        'feedback_2': likes2,
        'custom_answer_2': custom_answer2,
        'generated_answer_3': answer3,
        'context_3': context3,
        'feedback_3': likes3,
        'custom_answer_3': custom_answer3
    }]}
    # save to csv --> get question and answers here.
    filepath = "feedback.json"
    if os.path.exists(filepath):
        with open("feedback.json", "r+") as f:
            file_data = json.load(f)
            file_data['gradio_feedback'].append(new_data['gradio_feedback'][0])
            f.seek(0)
            json.dump(file_data, f, indent = 4)
    else:
        with open("feedback.json", "w") as f:
            json.dump(new_data, f)
    

def make_inference_id(name: str) -> str:
  '''
    🎯 Best practice to ensure unique Workflow names.
    '''
  from datetime import datetime

  import pytz

  # Timezones: US/{Pacific, Mountain, Central, Eastern}
  # All timezones `pytz.all_timezones`. Always use caution with timezones.
  curr_time = datetime.now(pytz.timezone('US/Central'))
  return f"{name}-{str(curr_time.strftime('%h_%d,%Y@%H:%M'))}"


if __name__ == '__main__':
  args = main_arg_parse()
  my_ta = TA_Gradio(args)
  # my_ta.model_evaluation()
  my_ta.main()

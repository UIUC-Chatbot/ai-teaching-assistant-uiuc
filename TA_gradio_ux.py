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

import main
import prompting
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

NUM_ANSWERS_GENERATED = 3
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

  def __init__(self, args):
    # dynamically select device based on available GPU memory
    self.device = torch.device(f'cuda:{get_device_with_most_free_memory()}')
    opt_device_list = get_gpu_ids_with_sufficient_memory(24)  # at least 24GB of memory

    self.ta = main.TA_Pipeline(opt_weight_path=args.model_weight,
                               ct2_path="../data/models/opt_acc/opt_1.3b_fp16",
                               is_server=True,
                               device_index_list=opt_device_list,
                               use_clip=True)
    self.prompter = prompting.Prompt_LLMs()
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

  def log_results_to_wandb(self, user_question, generated_answers_list, final_scores, top_context_list, user_defined_context,
                           runtime) -> None:
    wandb.log({'runtime (seconds)': runtime})

    results_table = wandb.Table(
        columns=["question", "user_supplied_context", "generated_answers", "retrieved_contexts", "scores", "runtime (seconds)"])
    for ans, score, retrieved_context in zip(generated_answers_list, final_scores, top_context_list):
      one_row_of_data = [user_question, user_defined_context, ans, retrieved_context, score, runtime]
      results_table.add_data(*one_row_of_data)

    # log a new table for each time our app is used. Can't figure out how to append to them easily.
    wandb.log({make_inference_id('Inference_made'): results_table})

  def add_gpt3_response(self,
                        results_df: pd.DataFrame,
                        user_question,
                        top_context_list: List[str],
                        use_equation_prompt: bool = False) -> pd.DataFrame:
    """
      GPT3 for comparison to SOTA.
      This answer is ALWAYS shown to the user, no matter the score. It's the first element in the dataframe. 
      It is scored by the ranker, but it is not subject to filtering like the other generations are.
    """
    generated_answer = self.ta.gpt3_completion(user_question, top_context_list[0], use_equation_prompt)
    score = self.ta.re_ranking_ms_marco([generated_answer], user_question)

    gpt3_result = {
        'Answer': [generated_answer],
        'Context': [top_context_list[0]],
        'Score': score,  # score is already a list
    }
    df_to_append = pd.DataFrame(gpt3_result)
    return pd.concat([df_to_append, results_df], ignore_index=True)

  def add_gpt3_fewshot_response(self, results_df: pd.DataFrame, user_question, top_context_list: List[str]) -> pd.DataFrame:
    """
    GPT3 few shot for comparison to SOTA.
    Note : few shot doesn't use context.
    This answer is ALWAYS shown to the user, no matter the score. It is not subject to score filtering like the other generations are.
    """
    generated_answer = "GPT-3 few-shot response:\n" + self.prompter.GPT3_fewshot(user_question)
    score = self.ta.re_ranking_ms_marco([generated_answer], user_question)
    gpt3_result = {
        'Answer': [generated_answer],
        'Context': [top_context_list[0]],  #context is not used in few shot answer generation
        'Score': score,  # score is already a list
    }
    df_to_append = pd.DataFrame(gpt3_result)
    return pd.concat([df_to_append, results_df], ignore_index=True)

  def load_text_answer(self, question, context, use_gpt3, use_equation_checkbox):
    '''
    This function is called when the user clicks the "Generate Answer" button.
    It collects responses and updates the gradio interface iteratively as we get new responses. 
    At the end, it shows a 'main answer' after all answers are generated AND ranked.
    '''
    # num_returns = 9 = 3 answers + 3 contexts + Gpt3 answer + final ranked answer + CLIP retrieval image list.
    NUM_RETURNS = 9
    # clear the previous answers if present
    clear_list = [gr.update(value=None) for _ in range(NUM_RETURNS)]
    clear_list[-1] = None  # CLIP image list
    print("CLEAR LIST: ", clear_list)
    yield clear_list

    # contexts
    top_context_list = self.ta.retrieve_contexts_from_pinecone(user_question=question, topk=NUM_ANSWERS_GENERATED)

    # GPT-3
    if use_gpt3:
      gpt3_generated_answer = self.ta.gpt3_completion(question, top_context_list[0], use_equation_checkbox)
      ans_list = [gr.update() for _ in range(NUM_RETURNS)]
      ans_list[-1] = None  # CLIP image value
      ans_list[-2] = gr.update(value=str(gpt3_generated_answer))
      yield ans_list
    else:
      gpt3_response = None

    # RUN CLIP -- todo, run right after GPT-3.
    ans_list = [gr.update() for _ in range(NUM_RETURNS)]
    #ans_list[-1] = self.run_clip(question)  # retrieved_images
    image_list = self.run_clip(question)
    ans_list[-1] = image_list
    yield ans_list

    # MAIN answer generation loop
    self.generated_answers_list = []
    for i, ans in enumerate(self.ta.yield_text_answer(question, context)):
      i = 2 * i
      ans_list = [gr.update() for _ in range(NUM_RETURNS)]
      ans_list[-1] = image_list  # CLIP image list

      ans_list[i] = gr.update(value=ans[0])
      ans_list[i + 1] = gr.update(value=ans[1])
      self.generated_answers_list.append(ans[0])
      yield ans_list

    # RANKING the answers here along with GPT-3 answer
    if gpt3_response is not None:
      self.generated_answers_list.append(gpt3_response[0])
      top_context_list.append(top_context_list[0])
    final_scores = self.ta.re_ranking_ms_marco(self.generated_answers_list, question)
    # print(final_scores)

    results = {
        'Answer': self.generated_answers_list,
        # append page number and textbook name to each context
        'Context': top_context_list,
        'Score': final_scores
    }
    print("RESULTS")
    print(len(results['Answer']))
    print(len(results['Context']))
    print(len(results['Score']))

    # this is causing errors. All arrays must be of the same length.
    generated_results_df = pd.DataFrame(results).sort_values(by=['Score'], ascending=False).head(NUM_ANSWERS_TO_SHOW_USER)
    ans_list = [gr.update() for _ in range(NUM_RETURNS)]
    ans_list[-1] = image_list  # CLIP image list

    # best answer is the 2nd last update
    generated_results_df = generated_results_df.reset_index()
    print("GENERATED RESULTS DF: ", generated_results_df)
    ans_list[-3] = gr.update(value=str(generated_results_df['Answer'][0]))
    yield ans_list

  def gpt3_textbox_visibility(use_gpt3):
    if use_gpt3:
      return gr.update(visible=True)
    else:
      return gr.update(visible=False)

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
          # top checkboxes
          with gr.Row():
            use_gpt3_checkbox = gr.Checkbox(label="Include GPT-3 (paid)?")
            use_equation_checkbox = gr.Checkbox(label="Prioritize equations?")
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
      ''' RESULTS SECTION for text answers '''
      with gr.Row():
        with gr.Column():
          gr.Markdown("""## Results""")
          best_answer = gr.Textbox(label="Best Answer", wrap=True)  # scroll_to_output=True
          gpt3_answer = gr.Textbox(label="GPT-3 Answer", wrap=True, visible=False)
          use_gpt3_checkbox.change(fn=self.gpt3_textbox_visibility, outputs=[gpt3_answer])

      with gr.Row():
        with gr.Column():
          generated_answer1 = gr.Textbox(label="Answer 1", wrap=True)
          context1 = gr.Textbox(label="Context 1", wrap=True)

          feedback_radio1 = gr.Radio(['Like', 'Dislike'], label="Feedback")
          custom_ans1 = gr.Textbox(label="What would the ideal answer have been?", input="text")
        with gr.Column():
          generated_answer2 = gr.Textbox(label="Answer 2", wrap=True)
          context2 = gr.Textbox(label="Context 2", wrap=True)

          feedback_radio2 = gr.Radio(['Like', 'Dislike'], label="Feedback")
          custom_ans2 = gr.Textbox(label="What would the ideal answer have been?", input="text")
        with gr.Column():
          generated_answer3 = gr.Textbox(label="Answer 3", wrap=True)
          context3 = gr.Textbox(label="Context 3", wrap=True)

          feedback_radio3 = gr.Radio(['Like', 'Dislike'], label="Feedback")
          custom_ans3 = gr.Textbox(label="What would the ideal answer have been?", input="text")

      with gr.Row():
        feedback_btn = gr.Button(value="Submit feedback")
        feedback_btn.click(save_feedback,
                           inputs=[
                               search_question, generated_answer1, context1, feedback_radio1, custom_ans1, generated_answer2, context2,
                               feedback_radio2, custom_ans2, generated_answer3, context3, feedback_radio3, custom_ans3
                           ],
                           outputs=[feedback_radio1, custom_ans1, feedback_radio2, custom_ans2, feedback_radio3, custom_ans3])

      # Show clip-retrieved images
      with gr.Row():
        with gr.Column():
          gr.Markdown("""## Lecture slides
                      We use two systems for image retrieval: standard CLIP and OCR + semantic search for text-heavy slides.
                      """)
          lec_gallery = gr.Gallery(label="Lecture images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")

        # event = run.click(fn=self.question_answer,
        #                   inputs=[search_question, context, use_gpt3_checkbox, image],
        #                   outputs=[generated_answer, lec_gallery],
        #                   scroll_to_output=True)

        run.click(
            fn=self.load_text_answer,
            inputs=[search_question, context, use_gpt3_checkbox, use_equation_checkbox],
            outputs=[
                generated_answer1,
                context1,
                generated_answer2,
                context2,
                generated_answer3,
                context3,
                best_answer,
                gpt3_answer,
                # TODO: add a gallery return here for the images.
                lec_gallery
            ])

    # ensure previous sessions are closed from our public port 8888
    gr.close_all()

    input_blocks.queue(concurrency_count=2)  # limit concurrency
    input_blocks.launch(share=True, favicon_path='./astro_on_horse.jpg')
    # input_blocks.launch(share=True, server_name='0.0.0.0', server_port=8888, favicon_path='./astro_on_horse.jpg')
    # debug=True
    # input_blocks.integrate(wandb=wandb)


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
  # save to json --> get question and answers here.
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

  # clear the feedback components
  clear_list = [gr.update(value=None) for i in range(6)]
  return clear_list


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

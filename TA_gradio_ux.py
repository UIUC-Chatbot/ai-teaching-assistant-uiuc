import os
import sys

ROOT_DIR = os.path.abspath("../retreival-generation-system/trt_accelerate/HuggingFace/")
sys.path.append(ROOT_DIR)
sys.path.append("../human_data_review")
sys.path.append("../retreival-generation-system")
sys.path.append("../retreival-generation-system/trt_accelerate")
import argparse
import pprint
import random
import time
from datetime import datetime
from typing import Dict, List

import gradio as gr
import main
import pandas as pd
import torch
import wandb
from PIL import Image
import json
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.evaluation.qa import QAEvalChain
from langchain.llms import OpenAI
import prompting

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from .autonotebook import tqdm as notebook_tqdm

# Todo: integrate CLIP.
# Todo: log images.
# wandb.log(
#     {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]},
#     step=0)


def main_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_entity', type=str, default='uiuc-ta-chatbot-team')
    parser.add_argument('--wandb_project', type=str, default="First_TA_Chatbot")
    # parser.add_argument('--trt_path',type = str, default= None)
    args = parser.parse_args()
    return args


import torch.autograd.profiler as profiler


class TA_Gradio():

    def __init__(self, args):
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        self.device = torch.device(args.device)
        self.ta = main.TA_Pipeline(
            device=self.device,
            opt_weight_path=args.model_weight,
            ct2_path = "../data/models/opt_acc/opt_1.3b_fp16",
            is_server = True,
            device_index = [0,3],
            n_stream = 2
            )  
        self.prompter = prompting.Prompt_LLMs()
        # accelerate OPT model (optimized model with multiple instances, parallel execution): 
        # ct2_path = "../data/models/opt_acc/opt_1.3b_fp16",
        # is_server = True,
        # device_index = [0,1],
        # n_stream = 3

    def run_clip(self, user_question: str, num_images_returned: int = 4):
        return self.ta.clip(user_question, num_images_returned)

    def model_evaluation(self, eval_set_path: str = '/home/zhiweny2/chatbotai/jerome/human_data_review/gpt-3_semantic_search/1_top_quality.json'):
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
    
    def question_answer(self, question: str, user_defined_context: str = '', use_gpt3: bool = False, image=None, use_equation_checkbox:bool = False):
        """
        This is the function called with the user clicks the main "Search 🔍" button.
        You can call this from anywhere to run our main program.
        
        question: user-supplied question
        [OPTIONAL] user_defined_context: user-supplied context to make the answer more specific. Usually it's empty, so we AI retrieve a context.
        [OPTIONAL] use_gpt3: Run GPT-3 answer-generation if True, default is False. The True/False value of the checkbox in the UI to "Use GPT3 (paid)". 
        [OPTIONAL] image: User-supplied image, for reverse image search.
        [OPTIONAL] use_equations : To include equations in the answer
        """
        start_time = time.monotonic()
        # we generate many answers, then filter it down to the best scoring ones (w/ msmarco).
        NUM_ANSWERS_GENERATED = 3 
        NUM_ANSWERS_TO_SHOW_USER = 3
        NUM_IMAGES_TO_SHOW_USER = 4  # 4 is good for gradio image layout
        USER_QUESTION = str(question)
        print("-----------------------------\nINPUT USER QUESTION:", USER_QUESTION, '\n-----------------------------')

        # check if user supplied their own context.
        if len(user_defined_context) == 0:
            # contriever: find relevant passages
            # top_context_list = self.ta.retrieve(
            #     user_question=USER_QUESTION,
            #     topk=NUM_ANSWERS_GENERATED)

            start_time_pinecone = time.monotonic()
            top_context_documents = self.ta.retrieve_contexts_from_pinecone(user_question=USER_QUESTION,
                                                                            topk=NUM_ANSWERS_GENERATED)
            top_context_metadata = [
                f"Source: page {int(doc.metadata['page_number'])} in {doc.metadata['textbook_name']}"
                for doc in top_context_documents
            ]
            top_context_list = [doc.page_content for doc in top_context_documents]
            print(f"⏰ Runtime for Pinecone: {(time.monotonic() - start_time_pinecone):.2f} seconds")
            # print(doc.metadata['page_number'], doc.metadata['textbook_name'])

            # TODO: add OPT back in when Wentao is ready.
            # Run opt answer generation
            generated_answers_list = self.ta.OPT(USER_QUESTION,
                                                 top_context_list,
                                                 NUM_ANSWERS_GENERATED,
                                                 print_answers_to_stdout=False)

            # T5 generations
            generated_answers_list.extend(self.ta.run_t5_completion(USER_QUESTION,
                                                               top_context_list,
                                                               num_answers_generated=NUM_ANSWERS_GENERATED,
                                                               print_answers_to_stdout=True))
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

        # rank potential answers
        # todo: rank both!!
        final_scores = self.ta.re_ranking_ms_marco(generated_answers_list[3:], USER_QUESTION)

        # return a pd datafarme, to display a gr.dataframe
        results = {
            'Answer': generated_answers_list[3:],
            # append page number and textbook name to each context
            'Context': [f"{text}. {meta}" for text, meta in zip(top_context_list, top_context_metadata)],
            # 'Context': top_context_list,
            'Score': final_scores,
        }
        print(len(generated_answers_list))
        print(len(top_context_list))
        print(len(final_scores))
        # sort results by MSMarco ranking
        generated_results_df = pd.DataFrame(results).sort_values(by=['Score'],
                                                                 ascending=False).head(NUM_ANSWERS_TO_SHOW_USER)

        # GPT3 for comparison to SOTA. Append to df to ensure it's ALWAYS displayed, regardless of msmarco score.
        if use_gpt3:
            generated_results_df = self.add_gpt3_response(generated_results_df, USER_QUESTION, top_context_list)

        # todo: include gpt3 results in logs. generated_results_df to wandb.
        # append data to wandb
        self.log_results_to_wandb(USER_QUESTION, generated_answers_list, final_scores, top_context_list,
                                  user_defined_context,
                                  time.monotonic() - start_time)

        # Flag for if we want to use CLIP or not.
        use_clip = False  # TODO: change this when I fix clip.
        if use_clip:
            return generated_results_df, self.run_clip(question, NUM_IMAGES_TO_SHOW_USER)
        else:
            # without running clip
            return generated_results_df, None

    def log_results_to_wandb(self, user_question, generated_answers_list, final_scores, top_context_list,
                             user_defined_context, runtime) -> None:
        wandb.log({'runtime (seconds)': runtime})

        results_table = wandb.Table(columns=[
            "question", "user_supplied_context", "generated_answers", "retrieved_contexts", "scores",
            "runtime (seconds)"
        ])
        for ans, score, retrieved_context in zip(generated_answers_list, final_scores, top_context_list):
            one_row_of_data = [user_question, user_defined_context, ans, retrieved_context, score, runtime]
            results_table.add_data(*one_row_of_data)

        # log a new table for each time our app is used. Can't figure out how to append to them easily.
        wandb.log({make_inference_id('Inference_made'): results_table})

    def add_gpt3_response(self, results_df: pd.DataFrame, user_question, top_context_list: List[str], use_equation_checkbox:bool=False) -> pd.DataFrame:
        """
        GPT3 for comparison to SOTA.
        This answer is ALWAYS shown to the user, no matter the score. It is not subject to score filtering like the other generations are.
        Change use_equation_checkbox = True to display equations
        """
        generated_prompt = self.prompter.prepare_prompt(user_question, top_context_list[0], use_equation_checkbox)
        generated_answer  = "GPT-3 response:\n" + self.prompter.GPT3_response_API(generated_prompt)
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
            'Context': [top_context_list[0]], #context is not used in few shot answer generation
            'Score': score,  # score is already a list
        }
        df_to_append = pd.DataFrame(gpt3_result)
        return pd.concat([df_to_append, results_df], ignore_index=True)

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
                    search_question = gr.Textbox(
                        label="Search\n",
                        placeholder="Ask me anything...",
                    )
                    context = gr.Textbox(
                        label="(Optional) give a relevant textbook paragraph for specific questions",
                        placeholder="(Optional) we'll use the paragraph to generate an answer to your question.")
                    # gr.Markdown("""Try searching for:""")
                    use_gpt3_checkbox = gr.Checkbox(label="Include GPT-3 (paid)?")
                    use_equation_checkbox = gr.Checkbox(label="Include relevant equations?")
                    examples = gr.Examples(
                        examples=[
                            ["What is a Finite State Machine?"],
                            ["How do you design a functional a Two-Bit Gray Code Counter?"],
                        ],
                        inputs=[search_question, context],  # todo: fix img part
                        outputs=[],
                    )
                # reverse image search
                image = gr.Image(type="pil",
                                 label="[NOT IMPLEMENTED YET] -- Reverse Image Search (optional)",
                                 shape=(224, 224))
            ''' Button and on-click function '''
            with gr.Row():
                # create a button with an orange background
                # run = gr.Button("Search 🔍", style='')
                run = gr.Button(
                    "Search  🔍",
                    variant='primary',
                )
                # run_reverse_img_search = gr.Button("Image search", variant='secondary',)
            ''' RESULTS SECTION, for text search && CLIP '''
            with gr.Row():
                gr.Markdown("""## Results""")

            event = run.click(fn=self.question_answer,
                              inputs=[search_question, context, use_gpt3_checkbox, image, use_equation_checkbox],
                              outputs=[
                                  gr.Dataframe(
                                      headers=["Answer", "Score", "Context", "Metadata"],
                                      wrap=True,
                                  ),
                                  gr.Gallery(label="Lecture images", show_label=False,
                                             elem_id="gallery").style(grid=[2], height="auto")
                              ],
                              scroll_to_output=True)

            with gr.Row():
                txt = gr.Textbox(label="chat", lines=2)
                chatbot = gr.Chatbot().style(color_map=("green", "pink"))
            with gr.Row():
                chat = gr.Button("Chat", variant='primary')

            event_chat = chat.click(
                self.chat,
                inputs=[txt, chatbot],
                outputs=[chatbot],
            )
            ''' Reverse image search '''
            # event_2 = run_reverse_img_search.click(
            #     fn=self.run_clip,
            #     inputs=[search_question],  # question, num_images_returned
            #     outputs=[gr.Gallery(
            #         label="Lecture images", show_label=False, elem_id="gallery"
            #         ).style(grid=[2], height="auto")],
            #     scroll_to_output=True)

        # demo = gr.Interface(
        #     self.chat,
        #     ["text", "state"],
        #     [chatbot, "state"],
        #     allow_flagging="never",
        # )
        input_blocks.queue(concurrency_count=2)  # limit concurrency
        input_blocks.launch(share=True)
        input_blocks.integrate(wandb=wandb)


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
    


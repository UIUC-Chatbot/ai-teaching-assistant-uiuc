import os
import sys

ROOT_DIR = os.path.abspath(
    "../retreival-generation-system/trt_accelerate/HuggingFace/")
sys.path.append(ROOT_DIR)
sys.path.append("../retreival-generation-system")
sys.path.append("../retreival-generation-system/trt_accelerate")
import argparse
import pprint
import random
import time
from typing import Dict, List

import gradio as gr
import main
import pandas as pd
import torch
import wandb
from PIL import Image

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
    parser.add_argument('--wandb_entity',
                        type=str,
                        default='uiuc-ta-chatbot-team')
    parser.add_argument('--wandb_project',
                        type=str,
                        default="First_TA_Chatbot")
    # parser.add_argument('--trt_path',type = str, default= None)
    args = parser.parse_args()
    return args


class TA_Gradio():

    def __init__(self, args):
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        self.device = torch.device(args.device)
        self.ta = main.TA_Pipeline(
            device=self.device,
            opt_weight_path=args.model_weight)  # ,trt_path=args.trt_path)

    def run_clip(self, user_question: str, num_images_returned: int = 4):
        return self.ta.clip(user_question, num_images_returned)

    def question_answer(self,
                        question: str,
                        user_defined_context: str,
                        use_gpt3: bool = False,
                        image=None):
        """
        This is the function called with the user clicks the main "Search 🔍" button.
        
        question: user-supplied question
        user_defined_context: user-supplied context, usually empty, so we retrieve our own.
        use_gpt3: the True/False value of the checkbox in the UI to "Use GPT3 (paid)"
        image: I'm not sure, I think this is the user-supplied image, for reverse image search.
        """
        # we generate many answers, then filter it down to the best scoring ones (w/ msmarco).
        NUM_ANSWERS_GENERATED = 5
        NUM_ANSWERS_TO_SHOW_USER = 3
        NUM_IMAGES_TO_SHOW_USER = 4  # 4 is good for gradio image layout

        USER_QUESTION = str(question)

        start_time = time.monotonic()

        # check if user supplied their own context.
        if len(user_defined_context) == 0:
            # contriever: find relevant passages
            top_context_list = self.ta.retrieve(
                user_question=USER_QUESTION,
                num_answers_generated=NUM_ANSWERS_GENERATED)

            # Run opt answer generation
            # generated_answers_list = self.ta.OPT(USER_QUESTION,
            #                                      top_context_list,
            #                                      NUM_ANSWERS_GENERATED,
            #                                      print_answers_to_stdout=False)

            # T5 generations
            generated_answers_list = self.ta.run_t5_completion(
                                                        USER_QUESTION,
                                                        top_context_list,
                                                        num_answers_generated=NUM_ANSWERS_GENERATED,
                                                        print_answers_to_stdout=True)
            print("generated_answers_list", generated_answers_list)
        else:
            # opt: passage + question --> answer
            # generated_answers_list = self.ta.OPT_one_question_multiple_answers(
            #     USER_QUESTION,
            #     user_defined_context,
            #     num_answers_generated=NUM_ANSWERS_GENERATED,
            #     print_answers_to_stdout=False)

            # T5 generations
            generated_answers_list = self.ta.run_t5_completion(
                USER_QUESTION,
                user_defined_context,
                num_answers_generated=NUM_ANSWERS_GENERATED,
                print_answers_to_stdout=True)

            # show (the same) user-supplied context for next to each generated answer.
            top_context_list = [user_defined_context] * NUM_ANSWERS_GENERATED

        # rank potential answers
        # todo: rank both!!
        final_scores = self.ta.re_ranking_ms_marco(generated_answers_list,
                                                   USER_QUESTION)

        # return a pd datafarme, to display a gr.dataframe
        results = {
            # 'Answer': generated_answers_list,
            'Answer': generated_answers_list,
            'Context': top_context_list,
            'Score': final_scores,
        }

        # sort results by MSMarco ranking
        generated_results_df = pd.DataFrame(results).sort_values(
            by=['Score'], ascending=False).head(NUM_ANSWERS_TO_SHOW_USER)

        # GPT3 for comparison to SOTA. Append to df to ensure it's ALWAYS displayed, regardless of msmarco score.
        if use_gpt3:
            generated_results_df = self.add_gpt3_response(
                generated_results_df, USER_QUESTION, top_context_list)

        # todo: include gpt3 results in logs.
        # append data to wandb
        self.log_results_to_wandb(USER_QUESTION, generated_answers_list,
                                  final_scores, top_context_list,
                                  user_defined_context,
                                  time.monotonic() - start_time)

        # Flag for if we want to use CLIP or not.
        use_clip = False  # TODO: change this when I fix clip.
        if use_clip:
            return generated_results_df, self.run_clip(
                question, NUM_IMAGES_TO_SHOW_USER)
        else:
            # without running clip
            return generated_results_df, None

    def log_results_to_wandb(self, user_question, generated_answers_list,
                             final_scores, top_context_list,
                             user_defined_context, runtime) -> None:
        wandb.log({'runtime (seconds)': runtime})

        results_table = wandb.Table(columns=[
            "question", "user_supplied_context", "generated_answers",
            "retrieved_contexts", "scores", "runtime (seconds)"
        ])
        for ans, score, retrieved_context in zip(generated_answers_list,
                                                 final_scores,
                                                 top_context_list):
            one_row_of_data = [
                user_question, user_defined_context, ans, retrieved_context,
                score, runtime
            ]
            results_table.add_data(*one_row_of_data)

        # log a new table for each time our app is used. Can't figure out how to append to them easily.
        wandb.log({make_inference_id('Inference_made'): results_table})

    def add_gpt3_response(self, results_df: pd.DataFrame, user_question,
                          top_context_list: List[str]) -> pd.DataFrame:
        """
        GPT3 for comparison to SOTA.
        This answer is ALWAYS shown to the user, no matter the score. It is not subject to score filtering like the other generations are.
        """
        generated_answer = "GPT-3 response:\n" + self.ta.gpt3_completion(
            user_question, top_context_list[0])

        score = self.ta.re_ranking_ms_marco([generated_answer], user_question)

        gpt3_result = {
            'Answer': [generated_answer],
            'Context': [top_context_list[0]],
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

    def main(self, ):
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
                        label=
                        "(Optional) give a relevant textbook paragraph for specific questions",
                        placeholder=
                        "(Optional) we'll use the paragraph to generate an answer to your question."
                    )
                    # gr.Markdown("""Try searching for:""")
                    use_gpt3_checkbox = gr.Checkbox(
                        label="Include GPT-3 (paid)?")
                    examples = gr.Examples(
                        examples=[
                            ["What is a Finite State Machine?"],
                            [
                                "How do you design a functional a Two-Bit Gray Code Counter?"
                            ],
                        ],
                        inputs=[search_question,
                                context],  # todo: fix img part
                        outputs=[],
                    )
                # reverse image search
                image = gr.Image(
                    type="pil",
                    label=
                    "[NOT IMPLEMENTED YET] -- Reverse Image Search (optional)",
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

            event = run.click(
                fn=self.question_answer,
                inputs=[search_question, context, use_gpt3_checkbox, image],
                outputs=[
                    gr.Dataframe(
                        headers=["Answer", "Score", "Context"],
                        wrap=True,
                    ),
                    gr.Gallery(label="Lecture images",
                               show_label=False,
                               elem_id="gallery").style(grid=[2],
                                                        height="auto")
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
        input_blocks.queue(concurrency_count=3)  # limit concurrency
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
    my_ta.main()
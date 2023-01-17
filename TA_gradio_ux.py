import os
import sys

ROOT_DIR = os.path.abspath("../retreival-generation-system/trt_accelerate/HuggingFace/")
sys.path.append(ROOT_DIR)
sys.path.append("../retreival-generation-system")
sys.path.append("../retreival-generation-system/trt_accelerate")
import time
import main
import gradio as gr
import random
import torch
import pandas as pd
from PIL import Image
import wandb
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from .autonotebook import tqdm as notebook_tqdm

# Todo: integrate CLIP.
# Todo: log images.
# wandb.log(
#     {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]},
#     step=0)

def main_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight',type = str,default = None)
    parser.add_argument('--device',type = str,default = 'cuda:0')
    parser.add_argument('--wandb_entity',type = str,default = 'uiuc-ta-chatbot-team')
    parser.add_argument('--wandb_project',type = str, default = "First_TA_Chatbot")
    # parser.add_argument('--trt_path',type = str, default= None)
    args = parser.parse_args()
    return args

class TA_Gradio():
    def __init__(self,args):
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        self.device = torch.device(args.device)
        self.results_table = wandb.Table(columns=["question", "user_supplied_context", "generated_answers", "retrieved_contexts", "scores", "runtime (seconds)"])
        self.ta = main.TA_Pipeline(device = self.device,opt_weight_path = args.model_weight) # ,trt_path=args.trt_path)
        
    def run_clip(self, user_question:str, num_images_returned: int = 4):
        return self.ta.clip(user_question, num_images_returned)

    def question_answer(self, question, context, image=None):
        start_time = time.monotonic()
        
        # init our Main() class -- all our models are class properties
        USER_QUESTION = str(question)
        context = str(context)
        NUM_ANSWERS_GENERATED = 5
        if len(str(context)) == 0:
            # contriever: find relevant passages
            top_context_list = self.ta.retrieve(user_question=USER_QUESTION, num_answers_generated=NUM_ANSWERS_GENERATED)
            generated_answers_list = self.ta.OPT(USER_QUESTION, top_context_list, NUM_ANSWERS_GENERATED, print_answers_to_stdout=False)
        else:
            # opt: passage + question --> answer
            generated_answers_list = self.ta.OPT_one_question_multiple_answers(USER_QUESTION, context, num_answers_generated=NUM_ANSWERS_GENERATED, print_answers_to_stdout=False)
            top_context_list = [context]*NUM_ANSWERS_GENERATED
        
        # rank OPT answers
        scores = self.ta.re_ranking_ms_marco(generated_answers_list)
        index_of_best_answer = torch.argmax(scores) # get best answer
        final_scores = [score.numpy()[0] for score in scores]
        # print("\n-------------------------------------------------------------\n")
        # print("Best answer 👇\n", generated_answers_list[index_of_best_answer])

        # append data to wandb
        wandb.log({'runtime (seconds)': time.monotonic() - start_time})
        end_time = time.monotonic() - start_time
        for ans,score,retrieved_context in zip(generated_answers_list, final_scores, top_context_list):
            one_row_of_data = [question, context, ans, retrieved_context, score, end_time]
            self.results_table.add_data(*one_row_of_data)
        wandb.log({"Full inputs and results": self.results_table})
        
        # return a pd datafarme, to display a gr.dataframe
        results = {
            'Answer' : generated_answers_list,
            'Context' : top_context_list,
            'Score' : final_scores,
        }
        # sorted(results["Answer"], key=lambda x: x['Score'])
        
        # my_df = pd.DataFrame({"question": question, "user_supplied_context": context, "generated_answers": generated_answers_list, "retrieved_contexts": top_context_list, "scores": final_scores, 'runtime (seconds)': time.monotonic() - start_time})
        # wandb.log({"Full inputs and results": my_df})
        
        return pd.DataFrame(results).sort_values(by=['Score'], ascending=False).head(3), self.run_clip(question, 4)
        # return pd.DataFrame(results).sort_values(by=['Score'], ascending=False).head(3), None

    def chat(self,message, history):
        history = history or []
        
        user_utter, topic, topic_history = self.ta.et_main(message)
        print("Topic:",topic)
        psg = self.ta.retrieve(user_utter,1)
        out_ans = self.ta.OPT(user_utter, psg, 1,  False)[0]
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
            flagging_dir='user-flagged-to-review',
            
            ''' Main user input section '''
            with gr.Row():
                with gr.Column(scale=2.6):
                    search_question = gr.Textbox(label="Search\n", placeholder="Ask me anything...",)
                    context = gr.Textbox(label="(Optional) give a relevant textbook paragraph for specific questions", placeholder="(Optional) we'll use the paragraph to generate an answer to your question.")
                    # gr.Markdown("""Try searching for:""")
                    examples=gr.Examples(
                        examples=[["What is a Finite State Machine?"],["How do you design a functional a Two-Bit Gray Code Counter?"],],
                        inputs=[search_question, context],  # todo: fix img part
                        outputs=[], )
                # reverse image search
                image = gr.Image(type="pil", label="[NOT IMPLEMENTED YET] -- Reverse Image Search (optional)", shape=(224, 224))
            
            ''' Button and on-click function '''
            with gr.Row(equal_height=True):
                # create a button with an orange background
                # run = gr.Button("Search 🔍", style='')
                run = gr.Button("Search  🔍", variant='primary',)
                # run_reverse_img_search = gr.Button("Image search", variant='secondary',)

            ''' RESULTS SECTION, for text search && CLIP '''
            with gr.Row(equal_height=True):
                gr.Markdown("""## Results""")
            
            event = run.click(
                fn=self.question_answer, 
                inputs=[search_question, context, image], 
                outputs=[gr.Dataframe(
                    headers=["Answer", "Score", "Context"],
                    wrap=True,
                    ), 
                    gr.Gallery(
                    label="Lecture images", show_label=False, elem_id="gallery"
                    ).style(grid=[2], height="auto")], 
                scroll_to_output=True)
            
            with gr.Row(equal_height=True):
                txt = gr.Textbox(label="chat", lines=2)
                chatbot = gr.Chatbot().style(color_map=("green", "pink"))
            with gr.Row(equal_height=True):
                chat = gr.Button("Chat",variant='primary')
            
            event_chat = chat.click(
                self.chat,
                inputs = [txt,chatbot],
                outputs = [chatbot],
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
        input_blocks.queue(concurrency_count=2) # limit concurrency
        input_blocks.launch(share=True)
        input_blocks.integrate(wandb=wandb)


if __name__ == '__main__':
    args = main_arg_parse()
    my_ta = TA_Gradio(args)
    my_ta.main()
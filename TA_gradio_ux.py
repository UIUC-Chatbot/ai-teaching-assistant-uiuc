import os
import time
import main
import gradio as gr
import torch
import pandas as pd
from PIL import Image
import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# from .autonotebook import tqdm as notebook_tqdm

# Todo: integrate CLIP.
# Todo: log images.
# wandb.log(
#     {"Style reference": [wandb.Image(transforms.ToPILImage()(target_im))]},
#     step=0)

class TA_Gradio():
    def __init__(self):
        wandb.init(project="First_TA_Chatbot", entity="kastan")
        self.results_table = wandb.Table(columns=["question", "user_supplied_context", "generated_answers", "retrieved_contexts", "scores", "runtime (seconds)"])
        self.ta = main.TA_Pipeline()

    def question_answer(self, question, context, image=None):
        start_time = time.monotonic()
        
        # init our Main() class -- all our models are class properties
        
        USER_QUESTION = str(question)
        context = str(context)
        NUM_ANSWERS_GENERATED = 10
        if len(str(context)) == 0:
            # contriever: find relevant passages
            top_context_list = self.ta.contriever(user_question=USER_QUESTION, num_answers_generated=NUM_ANSWERS_GENERATED)
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
        return pd.DataFrame(results).sort_values(by=['Score'], ascending=False)

    def main(self,):
        with gr.Blocks() as input_blocks:
            # title and description or use gr.HTML(...)
            gr.Markdown("""# Ask an Electrical Engineering Question
                        #### Our system will answer your question directly, and give links to all your course materials.
                        """)
            flagging_dir='user-flagged-to-review',
            
            ''' Main user input section '''
            with gr.Row(equal_height=True):
                with gr.Column(scale=2.6, equal_height=True):
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
                

            ''' RESULTS SECTION (below)'''
            with gr.Row(equal_height=True):
                gr.Markdown("""## Results""")
            event = run.click(
                fn=self.question_answer, 
                inputs=[search_question, context, image], 
                outputs=[gr.Dataframe(
                    headers=["Answer", "Score", "Contexts"],
                    wrap=True,
                    )], 
                scroll_to_output=True)

        input_blocks.queue(concurrency_count=2) # limit concurrency
        input_blocks.launch(share=True)
        input_blocks.integrate(wandb=wandb)

my_ta = TA_Gradio()
my_ta.main()
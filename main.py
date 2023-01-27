import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

sys.path.append("../data-generator")
sys.path.append("../info-retrieval")
sys.path.append("../info-retrieval/CLIP_for_PPTs")
sys.path.append("../retreival-generation-system")

# set huggingface cace to our base dir, so we all share it. 
os.environ['TRANSFORMERS_CACHE'] = '/mnt/project/chatbotai'

# for CLIP
import clip
# from docquery import document, pipeline   # import docquery
import contriever.contriever_final  # import Asmita's contriever
# for gpt-3 completions
import openai
import torch
from clip_for_ppts import \
    ClipImage  # Hiral's clip forward & reverse image search.
from entity_tracker import \
    entity_tracker  # import entity tracker(dialog management)
from module import *  # import generation model(OPT/T5)
from PIL import Image
# for re-ranking MS-Marco
# for OPT
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          T5ForConditionalGeneration, GPT2Tokenizer, OPTForCausalLM)


# question re-writing done, but we should use DST
# add re-ranker

########################
####  CHANGE ME 😁  ####
########################
NUM_ANSWERS_GENERATED = 5
# MAX_TEXT_LENGTH = 512
MAX_TEXT_LENGTH = 256
USER_QUESTION = ''


class TA_Pipeline:

    def __init__(self,
                 opt_weight_path=None,
                 trt_path=None,
                 device=torch.device("cuda:1"),
                 use_clip=False):

        # init parameters
        self.device = device
        self.opt_weight_path = opt_weight_path
        self.trt_path = trt_path
        self.LECTURE_SLIDES_DIR = os.path.join(os.getcwd(), "lecture_slides")

        # Retriever model: contriever
        self.contriever = None
        # Generation model: OPT & T5
        self.opt_model = None
        self.t5_model = None
        self.t5_tokenizer = None

        # Reranker
        self.rerank_msmarco_model = None
        self.rerank_msmarco_tokenizer = None
        # DocQuery pipeline
        self.pipeline = None
        self.doc = None
        # Entity tracker
        self.et = None
        # Clip for image search
        if use_clip:
            self.clip_search_class = None
            self._load_clip()

        # Load everything into cuda memory
        self.load_modules()

        # init to reasonable defaults (these will typically be overwritten when invoked)
        self.user_question = USER_QUESTION
        # self.num_answers_generated = NUM_ANSWERS_GENERATED
        self.max_text_length = MAX_TEXT_LENGTH

    ######################################################################
    ########  Load all our different models ##############################
    ######################################################################

    def load_modules(self):
        self._load_opt()
        self._load_reranking_ms_marco()
        self._load_contriever()
        self._load_et()
        self._load_t5()
        # TODO: install doc-query dependencies
        # self._load_doc_query()

    def _load_clip(self):
        print("initing clip model...")
        print("Todo: think more carefully about which device to use.")

        self.clip_search_class = ClipImage(
            path_of_ppt_folders=self.LECTURE_SLIDES_DIR,
            path_to_save_image_features=os.getcwd(),
            mode='text',
            device='cuda:1')

    def _load_contriever(self):
        self.contriever = contriever.contriever_final.ContrieverCB()

    def _load_et(self):
        self.et = entity_tracker('ECE120')

    def _load_opt(self):
        """ Load OPT model """

        # todo: is this the right way to instantiate this model?
        self.opt_model = opt_model(
            "facebook/opt-1.3b",
            device=self.device)  # trt_path = self.trt_path

        if (self.opt_weight_path != None and self.trt_path == None):
            self.opt_model.load_checkpoint(self.opt_weight_path)

    def _load_reranking_ms_marco(self):
        self.rerank_msmarco_model = AutoModelForSequenceClassification.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-6-v2').to(self.device)
        self.rerank_msmarco_tokenizer = AutoTokenizer.from_pretrained(
            'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.rerank_msmarco_model.eval()

    def _load_doc_query(self):
        self.pipeline = pipeline('document-question-answering')
        # self.doc = document.load_document("../data-generator/notes/Student_Notes_short.pdf") # faster runtime on short test doc.
        self.doc = document.load_document(
            "../data-generator/raw_data/notes/Student_Notes.pdf")

    
    ######################################################################
    ########  Start completion generators ################################
    ######################################################################

    def _load_t5(self):
        # TODO: use float32, from small experience it just produces nicer responses. Int8 is garbage from tim-deters.
        self.t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", torch_dtype=torch.float16)
        # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto", load_in_8bit=True)
        self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

    def run_t5_completion(self,
                          user_question: str = USER_QUESTION,
                          top_context_list: List = None,
                          num_answers_generated: int = NUM_ANSWERS_GENERATED,
                          print_answers_to_stdout: bool = False):
        """ Run T5 generator """
        start_time = time.monotonic()

        response_list = []
        assert num_answers_generated == len(top_context_list)
        for i in range(num_answers_generated):
            inner_time = time.monotonic()
            PROMPT = f"Answer the questions given the passage below. Context: {top_context_list[i]}. Question {user_question} Answer:"

            # todo: tune the correct cuda device number.
            inputs = self.t5_tokenizer(PROMPT, return_tensors="pt").to("cuda:1")
            outputs = self.t5_model.generate(**inputs, max_length=256, num_beams=3, early_stopping=True)
            single_answer = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            response_list.append(single_answer)
            print("Single answer:", single_answer)
        if print_answers_to_stdout:
            print(f"⏰ T5 runtime for SINGLE generation: {(time.monotonic() - inner_time):.2f} seconds")
            inner_time = time.monotonic()
        if print_answers_to_stdout:
            print(f"⏰ T5 runtime for num_generations = {num_answers_generated}: {(time.monotonic() - start_time):.2f} seconds")
            print("Generated Answers:")
            print(
                '\n---------------------------------NEXT---------------------------------\n'
                .join(response_list))
        return response_list

    def gpt3_completion(self,
                        question,
                        context,
                        model='text-davinci-003',
                        temp=0.7,
                        top_p=1.0,
                        tokens=1000,
                        freq_pen=0.0,
                        pres_pen=0.0) -> str:

        prompt = self.prepare_prompt(question, context)
        max_retry = 5
        retry = 0
        prompt = prompt.encode(
            encoding='utf-8',
            errors='ignore').decode()  # force it to fix any unicode errors
        while True:
            try:
                response = openai.Completion.create(model=model,
                                                    prompt=prompt,
                                                    temperature=temp,
                                                    max_tokens=tokens,
                                                    top_p=top_p,
                                                    frequency_penalty=freq_pen,
                                                    presence_penalty=pres_pen)
                text = response['choices'][0]['text'].strip()
                return text
            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return "GPT3 error: %s" % oops
                print('Error communicating with OpenAI:', oops)
                # todo: log to wandb.

    def et_main(self, user_utter):
        qr_user_utter, topic, history = self.et.main(user_utter)
        return qr_user_utter, topic, history

    def et_add_ans(self, answer: str):
        self.et.answer_attach(answer)

    def retrieve(self,
                 user_question: str,
                 num_answers_generated: int = NUM_ANSWERS_GENERATED):
        ''' Invoke contriever (with reasonable defaults).add()
    It finds relevant textbook passages for a given question.
    This can be used for prompting a generative model to generate an better/grounded answer.
    '''
        self.user_question = user_question
        self.num_answers_generated = num_answers_generated

        # print("User question: ", user_question)
        contriever_contexts = self.contriever.retrieve_topk(
            user_question,
            path_to_json="../data-generator/input_data/split_textbook/paragraphs.json",
            k=num_answers_generated)
        top_context_list = self._contriever_clean_contexts(
            list(contriever_contexts.values()))
        # print(top_context_list)

        return top_context_list

    def _contriever_clean_contexts(self, raw_context_list):
        ''' clean contriever results. Currently this only removed newline characters. That's the main problem. '''
        top_context_list = []
        for i, context in enumerate(raw_context_list):
            cleaned_words_list = []
            for sub in context:
                cleaned_words_list.append(sub.replace("\n", ""))
            top_context_list.append("".join(cleaned_words_list))

        return top_context_list

    def OPT(self,
            user_question: str = USER_QUESTION,
            top_context_list: List = None,
            num_answers_generated: int = NUM_ANSWERS_GENERATED,
            print_answers_to_stdout: bool = True):
        """ Run OPT """
        response_list = []
        assert num_answers_generated == len(top_context_list)
        for i in range(num_answers_generated):
            opt_answer = self.opt_model.answer_question(
                top_context_list[i], user_question, MAX_TEXT_LENGTH)
            response_list.append(opt_answer)
        if print_answers_to_stdout:
            print("Generated Answers:")
            print(
                '\n---------------------------------NEXT---------------------------------\n'
                .join(response_list))
        return response_list

    def OPT_one_question_multiple_answers(
            self,
            user_question: str = USER_QUESTION,
            context: str = '',
            num_answers_generated: int = NUM_ANSWERS_GENERATED,
            print_answers_to_stdout: bool = True):
        """ Run OPT """
        response_list = []
        for i in range(num_answers_generated):
            opt_answer = self.opt_model.answer_question(
                context, user_question, MAX_TEXT_LENGTH)
            response_list.append(opt_answer)

        if print_answers_to_stdout:
            print("Generated Answers:")
            print(
                '\n---------------------------------NEXT---------------------------------\n'
                .join(response_list))
        return response_list

    def re_ranking_ms_marco(self, response_list: List, user_question: str):
        features = self.rerank_msmarco_tokenizer(
            [user_question] * len(response_list),
            response_list,
            padding=True,
            truncation=True,
            return_tensors="pt").to(self.device)
        with torch.no_grad():
            scores = self.rerank_msmarco_model(**features).logits.cpu()

        # torch tensor to numpy
        return [score.numpy()[0] for score in scores]

    def doc_query(self, user_question, num_answers_generated: int = 3):
        """ Run DocQuery. Lots of extra dependeicies. 
        TODO: make it so we can save the 'self.doc' object to disk and load it later.
        """
        self.user_question = user_question
        answer = self.pipeline(question=self.user_question,
                               **self.doc.context,
                               top_k=num_answers_generated)
        # todo: this has page numbers, that's nice.
        return answer[0]['answer']

    def clip(self, search_question: str, num_images_returned: int = 3):
        """ Run CLIP. 
    Returns a list of images in all cases. 
    """
        imgs = self.clip_search_class.text_to_image_search(
            search_text=search_question, top_k_to_return=num_images_returned)

        img_path_list = []
        for img in imgs:
            img_path_list.append(
                os.path.join(self.LECTURE_SLIDES_DIR, img[0], img[1]))
        print("Final image path: ", img_path_list)

        return img_path_list

    def prepare_prompt(self, question: str, context: str) -> str:
        """prepares prompt based on type of question - factoid, causal or listing"""
        factoid = ["What", "Where", "When", "Explain", "Discuss", "Clarify"]
        causal = ["Why", "How"]
        listing = ["List", "Break down"]
        if any(word in question for word in factoid):
            prompt = """Generate an objective, formal and logically sound answer to this question, based on the given context. 
            The answer must spur curiosity, enable interactive discussions and make the user ask further questions. 
            It should be interesting and use advanced vocabulary and complex sentence structures.
            Context : """ + context.replace(
                "\n", " ") + "\nQuestion:" + question.replace(
                    "\n", " ") + "\nAnswer:"
        elif any(word in question for word in causal):
            prompt = """Generate a procedural, knowledgeable and reasoning-based answer about this question, based on the given context. 
            The answer must use inference mechanisms and logic to subjectively discuss the topic. It should be creative and logic-oriented, analytical and extensive. Context :""" + context.replace(
                "\n", " ") + "\nQuestion:" + question.replace(
                    "\n", " ") + "\nAnswer:"
        elif any(word in question for word in listing):
            prompt = """Generate a list-type, descriptive answer to this question, based on the given context. 
            The answer should be very detailed and contain reasons, explanations and elaborations about the topic. It should be interesting and use advanced vocabulary and complex sentence structures. Context :""" + context.replace(
                "\n", " ") + "\nQuestion:" + question.replace(
                    "\n", " ") + "\nAnswer:"
        else:
            prompt = """Generate a detailed, interesting answer to this question, based on the given context. 
            The answer must be engaging and provoke interactions. It should use academic language and a formal tone. 
            Context : """ + context.replace(
                "\n", " ") + "\nQuestion:" + question.replace(
                    "\n", " ") + "\nAnswer:"
        return prompt

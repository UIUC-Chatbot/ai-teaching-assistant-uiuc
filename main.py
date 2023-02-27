# import ray
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

import pinecone  # cloud-hosted vector database for context retrieval
# for vector search
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone

# for auto-gpu selection
from gpu_memory_utils import (get_device_with_most_free_memory, get_free_memory_dict, get_gpu_ids_with_sufficient_memory)

sys.path.append("../data-generator")
sys.path.append("../info-retrieval")
sys.path.append("../info-retrieval/CLIP_for_PPTs")
sys.path.append("../retreival-generation-system")
import prompting

# set huggingface cace to our base dir, so we all share it.
os.environ['TRANSFORMERS_CACHE'] = '/mnt/project/chatbotai/huggingface_cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/mnt/project/chatbotai/huggingface_cache/datasets'

# for CLIP
# import clip
# from docquery import document, pipeline   # import docquery
import contriever.contriever_final  # import Asmita's contriever
# for gpt-3 completions
import openai
import torch
# Hiral's clip forward & reverse image search.
from clip_for_ppts import ClipImage
# import entity tracker(dialog management)
from entity_tracker import entity_tracker
# for OPT
from module import *  # import generation model(OPT/T5)
from PIL import Image
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer, OPTForCausalLM, T5ForConditionalGeneration)


class TA_Pipeline:

  def __init__(self,
               opt_weight_path=None,
               trt_path=None,
               ct2_path=None,
               is_server=False,
               device_index_list=None,
               device=torch.device(f"cuda:{get_device_with_most_free_memory()}"),
               use_clip=False):

    # init parameters
    self.user_question = ''
    self.max_text_length = None
    self.pinecone_index_name = 'uiuc-chatbot'  # uiuc-chatbot-v2

    # init parameters
    self.device = device
    self.opt_weight_path = opt_weight_path
    self.num_answers_generated = 3

    # OPT acceleration
    self.trt_path = trt_path
    self.ct2_path = ct2_path
    self.is_server = is_server
    self.device_index = device_index_list
    self.n_stream = len(device_index_list)

    # Reranker
    self.rerank_msmarco_model = None
    self.rerank_msmarco_tokenizer = None
    self.rerank_msmarco_device = None
    # josh's version of this model
    # self.rerank_msmarco_model = AutoModelForSequenceClassification.from_pretrained('/home/jkmin3/chatbotai/josh/data-generator/ranking_models/fine_tuning_MSmarco/final')
    # self.rerank_msmarco_tokenizer = AutoTokenizer.from_pretrained('/home/jkmin3/chatbotai/josh/data-generator/ranking_models/fine_tuning_MSmarco/final')

    # DocQuery pipeline
    self.pipeline = None
    self.doc = None
    # Entity tracker
    self.et = None
    # Pinecone vector store (for relevant contexts)
    self.vectorstore = None
    # Clip for image search
    self.LECTURE_SLIDES_DIR = os.path.join(os.getcwd(), "lecture_slides")
    if use_clip:
      self.clip_search_class = None
      self._load_clip()

    # Retriever model: contriever
    self.contriever = None
    # Generation model: OPT & T5
    self.opt_model = None
    self.t5_model = None
    self.t5_tokenizer = None

    #prompting
    self.prompter = prompting.Prompt_LLMs()

    # Clip for image search
    if use_clip:
      self.clip_search_class = None
      self._load_clip()

    # Load everything into cuda memory
    self.load_modules()

  ######################################################################
  ########  Load all our different models ##############################
  ######################################################################

  def yield_text_answer(
      self,
      user_question: str = '',
      user_defined_context: str = '',
  ):
    '''
    This is called by the Gradio app to yeild completions. Right now it only calls T5, would be best to have OPT, too.
    '''
    if user_defined_context:
      top_context_list = [user_defined_context * self.num_answers_generated]
    else:
      top_context_list = self.retrieve_contexts_from_pinecone(user_question=user_question, topk=self.num_answers_generated)

    for i, ans in enumerate(
        self.run_t5_completion(user_question=user_question,
                               top_context_list=top_context_list,
                               num_answers_generated=self.num_answers_generated,
                               print_answers_to_stdout=False)):
      yield ans, top_context_list[i]

  # def yield_text_answer(
  #   self,
  #   user_question: str = '',
  #   user_defined_context: str = ''):

  # if user_defined_context:
  #   top_context_list = [user_defined_context * self.num_answers_generated]
  # else:
  #   top_context_list = self.retrieve_contexts_from_pinecone(user_question=user_question, topk=self.num_answers_generated)

  def load_modules(self):
    # self._load_opt()
    # self._load_et()
    self._load_reranking_ms_marco()
    self._load_contriever()
    self._load_t5()
    self._load_pinecone_vectorstore()
    # TODO: install doc-query dependencies
    # self._load_doc_query()

  def _load_clip(self):
    print("initing clip model...")
    print("Todo: think more carefully about which device to use.")

    self.clip_search_class = ClipImage(path_of_ppt_folders=self.LECTURE_SLIDES_DIR,
                                       path_to_save_image_features=os.getcwd(),
                                       mode='text',
                                       device='cuda:1')

  def _load_contriever(self):
    self.contriever = contriever.contriever_final.ContrieverCB()

  def _load_et(self):
    self.et = entity_tracker('ECE120')

  def _load_opt(self):
    """ Load OPT model """
    # multiple instances
    self.opt_model = opt_model("facebook/opt-1.3b",
                               ct2_path=self.ct2_path,
                               device=self.device,
                               is_server=self.is_server,
                               device_index=self.device_index,
                               n_stream=self.n_stream)

    if (self.opt_weight_path != None and self.trt_path == None):
      self.opt_model.load_checkpoint(self.opt_weight_path)

  def _load_reranking_ms_marco(self):
    '''
    The fine-tuned ranking model from Josh: 
    AutoModelForSequenceClassification.from_pretrained('../data-generator/ranking_models/fine_tuning_MSmarco/cross-encoder-ms-marco-MiniLM-L-6-v2-2022-11-27_00-59-17').to(get_device_with_most_free_memory())
    '''
    self.rerank_msmarco_device = get_device_with_most_free_memory()
    self.rerank_msmarco_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2').to(
        self.rerank_msmarco_device)
    self.rerank_msmarco_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    self.rerank_msmarco_model.eval()

  def _load_doc_query(self):
    self.pipeline = pipeline('document-question-answering')
    # self.doc = document.load_document("../data-generator/notes/Student_Notes_short.pdf") # faster runtime on short test doc.
    self.doc = document.load_document("../data-generator/raw_data/notes/Student_Notes.pdf")

  def _load_t5(self):
    self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")

    # for now, device 2 is set to 0 because hongyu2 is running things there.
    self.t5_model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-xxl",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    #max_memory=get_free_memory_dict())
    # self.t5_model = torch.compile(self.t5_model) # no real speedup :(
  def t5(self, text):
      # todo: tune the correct cuda device number.
      inputs = self.t5_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda:0")
      outputs = self.t5_model.generate(**inputs,
                                          max_new_tokens=256,
                                          num_beams=3,
                                          early_stopping=True,
                                          temperature=1.5,
                                          repetition_penalty=2.5)
      single_answer = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
      return single_answer
      
  def T5_fewshot(self, user_question: str , top_context_list: List = None, num_answers_generated: int = None, print_answers_to_stdout: bool = False):
      """ Run T5 generator -few shot prompting """
      if num_answers_generated is None:
        num_answers_generated= self.num_answers_generated
      response_list = []
      assert num_answers_generated == len(
          top_context_list), "There must be a unique context for each generated answer. "
      for i in range(num_answers_generated):
          examples = """
          Task: Open book QA. Question: How do I check for overflow in a 2's complement operation. Answer: Overflow can be indicated in a 2's complement if the result has the wrong sign, such as if 2 positive numbers sum to a negative number or if 2 negative numbers sum to positive numbers.
          Task: Open book QA. Question: What is the order of precedence in C programming? Answer: PEMDAS (Parenthesis, Exponents, Multiplication, Division, Addition, Subtraction)
          Task: Open book QA. Question: Why would I use a constructive construct in C? Answer: A conditional construct would be used in C when you want a section of code to make decisions about what to execute based on certain conditions specified by you in the code. 
          """
          new_shot = examples + "Task: Open book QA. Question: %s \nContext : %s \nAnswer : " % (user_question, top_context_list[i])
          single_answer = self.t5(new_shot)
          response_list.append(single_answer)
      return response_list

  def gpt3_completion(self,
                      question,
                      context,
                      equation:bool = False, cot:bool = False,
                      model='text-davinci-003',
                      temp=0.7,
                      top_p=1.0,
                      tokens=1000,
                      freq_pen=1.0,
                      pres_pen=0.0) -> str:
      """ run gpt-3 for SOTA comparision, without few-shot prompting
      question : user_question 
      context : retrieved context
      [OPTIONAL] : equation flag to include equations
      [OPTIONAL] : chain-of-thought triggered by "Let's think step by step."
      """
      prompt = self.prompter.prepare_prompt(question, context, equation, cot)
      max_retry = 5
      retry = 0
      prompt = prompt.encode(encoding='utf-8', errors='ignore').decode()  # force it to fix any unicode errors
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


  def run_t5_completion(self,
                        user_question: str = '',
                        top_context_list: List = None,
                        num_answers_generated: int = None,
                        print_answers_to_stdout: bool = False):
    """ Run T5 generator """
    start_time = time.monotonic()

    if num_answers_generated is None:
      num_answers_generated = self.num_answers_generated

    response_list = []
    assert num_answers_generated == len(top_context_list), "There must be a unique context for each generated answer. "

    for i in range(num_answers_generated):
      inner_time = time.monotonic()
      PROMPT = f"Task: Open book QA. Question: {user_question} Context: {top_context_list[i]}. Answer:"
      single_answer = self.t5(PROMPT)
      response_list.append(single_answer)
      yield single_answer

      if print_answers_to_stdout:
        # print("Single answer:", single_answer)
        print(f"⏰ T5 runtime for SINGLE generation: {(time.monotonic() - inner_time):.2f} seconds")
        inner_time = time.monotonic()
    if print_answers_to_stdout:
      print(f"⏰ T5 runtime for {num_answers_generated} iters: {(time.monotonic() - start_time):.2f} seconds")
      print("Generated Answers:")
      print('\n---------------------------------NEXT---------------------------------\n'.join(response_list))
    # return response_list


  def et_main(self, user_utter):
    qr_user_utter, topic, history = self.et.main(user_utter)
    return qr_user_utter, topic, history

  def et_add_ans(self, answer: str):
    self.et.answer_attach(answer)

  ############################################################################
  ######### Context Retrieval (several types) ################################
  ############################################################################

  def _load_pinecone_vectorstore(self,):
    model_name = "intfloat/e5-large"  # best text embedding model. 1024 dims.
    pincecone_index = pinecone.Index("uiuc-chatbot")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment="us-west1-gcp")
    self.vectorstore = Pinecone(index=pincecone_index, embedding_function=embeddings.embed_query, text_key="text")

  def retrieve_contexts_from_pinecone(self, user_question: str, topk: int = None) -> List[Any]:
    ''' 
    Invoke Pinecone for vector search. These vector databases are created in the notebook `data_formatting_patel.ipynb` and `data_formatting_student_notes.ipynb`.
    Returns a list of LangChain Documents. They have properties: `doc.page_content`: str, doc.metadata['page_number']: int, doc.metadata['textbook_name']: str.
    '''
    if topk is None:
      topk = self.num_answers_generated

    # similarity search
    top_context_list = self.vectorstore.similarity_search(user_question, k=topk)

    # add the source info to the bottom of the context.
    top_context_metadata = [
        f"Source: page {int(doc.metadata['page_number'])} in {doc.metadata['textbook_name']}" for doc in top_context_list
    ]
    relevant_context_list = [f"{text}. {meta}" for text, meta in zip(top_context_list, top_context_metadata)]
    return relevant_context_list

  def retrieve(self, user_question: str, topk: int = None):
    ''' Invoke contriever (with reasonable defaults).add()
    It finds relevant textbook passages for a given question.
    This can be used for prompting a generative model to generate an better/grounded answer.
    '''
    self.user_question = user_question
    if topk is None:
      topk = self.num_answers_generated

    contriever_contexts = self.contriever.retrieve_topk(user_question,
                                                        path_to_json="../data-generator/input_data/split_textbook/paragraphs.json",
                                                        k=topk)
    top_context_list = self._contriever_clean_contexts(list(contriever_contexts.values()))

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
          user_question: str = '',
          top_context_list: List = None,
          num_answers_generated: int = None,
          print_answers_to_stdout: bool = True):
    """ Run OPT """

    if num_answers_generated is None:
      num_answers_generated = self.num_answers_generated

    response_list = []
    assert num_answers_generated == len(top_context_list)
    max_text_length = 256
    response_list = self.opt_model.answer_question_all(top_context_list, user_question, num_answers_generated, max_text_length)

    if print_answers_to_stdout:
      print("Generated Answers:")
      print('\n---------------------------------NEXT---------------------------------\n'.join(response_list))
    return response_list

  def OPT_one_question_multiple_answers(self,
                                        user_question: str = '',
                                        context: str = '',
                                        num_answers_generated: int = None,
                                        print_answers_to_stdout: bool = True):
    if num_answers_generated is None:
      num_answers_generated = self.num_answers_generated
    """ Run OPT """
    response_list = []
    max_text_length = 256
    for i in range(num_answers_generated):
      opt_answer = self.opt_model.answer_question(context, user_question, max_text_length)
      response_list.append(opt_answer)

    if print_answers_to_stdout:
      print("Generated Answers:")
      print('\n---------------------------------NEXT---------------------------------\n'.join(response_list))
    return response_list

  def re_ranking_ms_marco(self, response_list: List, user_question: str):
    features = self.rerank_msmarco_tokenizer([user_question] * len(response_list),
                                             response_list,
                                             padding=True,
                                             truncation=True,
                                             return_tensors="pt").to(self.rerank_msmarco_device)
    with torch.no_grad():
      scores = self.rerank_msmarco_model(**features).logits.cpu()

    # torch tensor to numpy
    return [score.numpy()[0] for score in scores]

  def doc_query(self, user_question, num_answers_generated: int = 3):
    """ Run DocQuery. Lots of extra dependeicies. 
        TODO: make it so we can save the 'self.doc' object to disk and load it later.
        """
    self.user_question = user_question
    answer = self.pipeline(question=self.user_question, **self.doc.context, top_k=num_answers_generated)
    # todo: this has page numbers, that's nice.
    return answer[0]['answer']

  def clip(self, search_question: str, num_images_returned: int = 3):
    """ Run CLIP. 
    Returns a list of images in all cases. 
    """
    imgs = self.clip_search_class.text_to_image_search(search_text=search_question, top_k_to_return=num_images_returned)

    img_path_list = []
    for img in imgs:
      img_path_list.append(os.path.join(self.LECTURE_SLIDES_DIR, img[0], img[1]))
    print("Final image path: ", img_path_list)

    return img_path_list


import sys
import os
import json
from typing import List, Dict, Any
import pathlib
sys.path.append("../data-generator")
sys.path.append("../info-retrieval")
sys.path.append("../info-retrieval/CLIP_for_PPTs")
sys.path.append("../retreival-generation-system")

# set environment variable huggingface cache path to ~/
os.environ['TRANSFORMERS_CACHE'] = '/home/kastanday/project'


# from docquery import document, pipeline   # import docquery
import contriever.contriever_final  # import Asmita's contriever
from module import *                # import generation model(OPT/T5)
from entity_tracker import entity_tracker # import entity tracker(dialog management)
from clip_for_ppts import ClipImage # Hiral's clip forward & reverse image search.
import torch

# question re-writing done, but we should use DST
# add re-ranker

# for OPT
from transformers import GPT2Tokenizer, OPTForCausalLM

# for re-ranking MS-Marco
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# for CLIP
import clip
from PIL import Image

########################
####  CHANGE ME 😁  ####
########################
NUM_ANSWERS_GENERATED = 5
# MAX_TEXT_LENGTH = 512
MAX_TEXT_LENGTH = 256
USER_QUESTION = ''

class TA_Pipeline:
  def __init__(self,opt_weight_path=None,trt_path = None,device = torch.device("cuda:0")):
    
    # init parameters
    self.device = device 
    self.opt_weight_path = opt_weight_path
    self.trt_path = trt_path
    self.LECTURE_SLIDES_DIR = os.path.join(os.getcwd(), "lecture_slides")
    
    # Retriever model: contriever
    self.contriever = None
    # Generation model: OPT
    self.opt_model = None 
    # Reranker
    self.rerank_msmarco_model = None 
    self.rerank_msmarco_tokenizer = None 
    # DocQuery pipeline 
    self.pipeline = None 
    self.doc = None
    # Entity tracker
    self.et = None 
    # Clip for image search
    self.clip_search_class = None 
    
    # Load everything into cuda memory    
    self.load_modules()
    
    # init to reasonable defaults (these will typically be overwritten when invoked)
    self.user_question = USER_QUESTION
    # self.num_answers_generated = NUM_ANSWERS_GENERATED
    self.max_text_length = MAX_TEXT_LENGTH
    
    self.clip_is_initted = False
  
  def load_modules(self):
    self._load_opt()
    self._load_reranking_ms_marco()
    # self._load_doc_query()
    self._load_contriever()
    self._load_et()
    self._load_clip()

  def _load_clip(self):
    print("initing clip model...")
    print("Todo: think more carefully about which device to use.")
    
    self.clip_search_class = ClipImage(path_of_ppt_folders=self.LECTURE_SLIDES_DIR,
                                       path_to_save_image_features=os.getcwd(),
                                       mode='text',
                                       device='cuda:1')
    
  def _load_contriever(self):
    self.contriever =contriever.contriever_final.ContrieverCB()
  
  def _load_et(self):
    self.et = entity_tracker('ECE120')
    
  def _load_opt(self):
    """ Load OPT model """
    self.opt_model = opt_model("facebook/opt-1.3b" ,device = self.device) # trt_path = self.trt_path
    if(self.opt_weight_path!=None and self.trt_path == None):
      self.opt_model.load_checkpoint(self.opt_weight_path)
    
  def _load_reranking_ms_marco(self):
    self.rerank_msmarco_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2').to(self.device)
    self.rerank_msmarco_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

  def _load_doc_query(self):
    self.pipeline = pipeline('document-question-answering')
    # self.doc = document.load_document("../data-generator/notes/Student_Notes_short.pdf") # faster runtime
    self.doc = document.load_document("../data-generator/notes/Student Notes.pdf")
  
  def et_main(self,user_utter):
    qr_user_utter, topic, history = self.et.main(user_utter)
    return qr_user_utter,topic,history
  
  def et_add_ans(self,answer:str):
    self.et.answer_attach(answer)
    
  def retrieve(self, user_question: str, num_answers_generated: int = NUM_ANSWERS_GENERATED):
    ''' Invoke contriever (with reasonable defaults).add()
    It finds relevant textbook passages for a given question.
    This can be used for prompting a generative model to generate an better/grounded answer.
    '''
    self.user_question = user_question
    self.num_answers_generated = num_answers_generated
    
    print("User question: ", user_question)
    contriever_contexts = self.contriever.retrieve_topk(user_question, path_to_json = "../data-generator/split_textbook/paragraphs.json", k = num_answers_generated)
    top_context_list = self._contriever_clean_contexts(list(contriever_contexts.values()))
    print(top_context_list)
    
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
  
  def OPT(self, user_question: str = USER_QUESTION, top_context_list: List = None, num_answers_generated: int = NUM_ANSWERS_GENERATED,  print_answers_to_stdout: bool = True):
    """ Run OPT """
    response_list = []
    assert num_answers_generated == len(top_context_list)
    for i in range(num_answers_generated):
      opt_answer = self.opt_model.answer_question(top_context_list[i],user_question,MAX_TEXT_LENGTH)
      response_list.append(opt_answer)
    if print_answers_to_stdout:
      print("Generated Answers:")
      print('\n---------------------------------NEXT---------------------------------\n'.join(response_list))
    return response_list
    
  def OPT_one_question_multiple_answers(self, user_question: str = USER_QUESTION, context: str = '', num_answers_generated: int = NUM_ANSWERS_GENERATED, print_answers_to_stdout: bool = True):
    """ Run OPT """
    response_list = []
    for i in range(num_answers_generated):
      opt_answer = self.opt_model.answer_question(context,user_question,MAX_TEXT_LENGTH)
      response_list.append(opt_answer)

    if print_answers_to_stdout:
      print("Generated Answers:")
      print('\n---------------------------------NEXT---------------------------------\n'.join(response_list))
    return response_list
    
  def re_ranking_ms_marco(self, response_list: List):
    features = self.rerank_msmarco_tokenizer([USER_QUESTION] * len(response_list), response_list,  padding=True, truncation=True, return_tensors="pt").to(self.device)
    self.rerank_msmarco_model.eval()
    with torch.no_grad():
        scores = self.rerank_msmarco_model(**features).logits.cpu()
        print("Scores for each answer (from ms_marco):", scores)
    return scores

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
    
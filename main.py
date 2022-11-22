import sys
import os
import json
from typing import List, Dict, Any
sys.path.append("../data-generator")
sys.path.append("../info-retrieval")
sys.path.append("../retreival-generation-system")

from docquery import document, pipeline   # import docquery
import contriever.contriever_final  # import Asmita's contriever
from module import *                # import generation model(OPT/T5)
from entity_tracker import entity_tracker # import entity tracker(dialog management)
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
MAX_TEXT_LENGTH = 300
USER_QUESTION = ''

class TA_Pipeline:
  def __init__(self,opt_weight_path=None,device = torch.device("cuda:0")):
    self.device = device 
    self.opt_weight_path = opt_weight_path
    
    # init modules 
    self._load_opt()
    self._load_reranking_ms_marco()
    self._load_doc_query()
    self._load_contriever()
    self._load_et()
    
    # init to reasonable defaults (these will typically be overwritten when invoked)
    self.user_question = USER_QUESTION
    # self.num_answers_generated = NUM_ANSWERS_GENERATED
    self.max_text_length = MAX_TEXT_LENGTH
    
    self.contriever_is_initted = True # todo: load contriever better?
    self.opt_is_initted = True
    self.ms_marco_is_initted = True
    self.doc_query_is_initted = True
    self.clip_is_initted = False

  def _load_contriever(self):
    self.contriever =contriever.contriever_final.ContrieverCB()
  
  def _load_et(self):
    self.et = entity_tracker('ECE120')
    
  def _load_opt(self):
    """ Load OPT model """
    self.opt_model = opt_model("facebook/opt-1.3b",self.device)
    if(self.opt_weight_path!=None):
      self.opt_model.load_checkpoint(self.opt_weight_path)
    self.opt_is_initted = True
    
  def _load_reranking_ms_marco(self):
    self.rerank_msmarco_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    self.rerank_msmarco_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    self.ms_marco_is_initted = True

  def _load_doc_query(self):
    self.pipeline = pipeline('document-question-answering')
    # self.doc = document.load_document("../data-generator/notes/Student_Notes_short.pdf") # faster runtime
    self.doc = document.load_document("../data-generator/notes/Student_Notes.pdf")
    self.doc_query_is_initted = True
  
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
    features = self.rerank_msmarco_tokenizer([USER_QUESTION] * len(response_list), response_list,  padding=True, truncation=True, return_tensors="pt")
    self.rerank_msmarco_model.eval()
    with torch.no_grad():
        scores = self.rerank_msmarco_model(**features).logits
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
    """ Run CLIP. """
    if not self.clip_is_initted:
      print("initing clip model...")
      self._load_clip()
    else:
      print("NOT initting clip")
      
    # Prepare the inputs
    SLIDES_DIR = "lecture_slides"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ppts = list(os.listdir(SLIDES_DIR))
    #print(ppts, len(ppts))
    text_inputs = torch.cat([clip.tokenize(search_question)]).to(device)
    res = []
    for i in ppts:
      #print(i)
      imgs = list(os.listdir(SLIDES_DIR+i))
      image_input = torch.cat([clip_preprocess(Image.open(SLIDES_DIR+i+'/'+image)).unsqueeze(0) for image in imgs]).to(device)

      with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

      # Pick the top 3 most similar labels for the image
      image_features /= image_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
      values, indices = similarity[0].topk(3)

      for val, index in zip(values,indices):
        # print(f"Image Name:{imgs[index]}\tSimilarity:{val}")
        res.append([i, imgs[index], val])
    
    # ans should have no of folders * 3 slides 
    ans = sorted(res,key=lambda x:x[2], reverse=True)
    print(ans[:3])
    
    img_list_to_return = []
    for i in range(num_images_returned):
      img_list_to_return.append(Image.open(SLIDES_DIR+ans[i][0]+"/"+ans[i][1]))
    return img_list_to_return
    
  
  def _load_clip(self):
    global clip_model
    global clip_preprocess
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on: ", device)
    clip_model, clip_preprocess = clip.load('ViT-B/32', device)
    self.clip_is_initted = True

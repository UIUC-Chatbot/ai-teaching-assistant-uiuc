import sys
import os
import json
sys.path.append("../data-generator")
sys.path.append("../info-retrieval")
import contriever.contriever_final # Asmita's contriever

# for OPT
from transformers import GPT2Tokenizer, OPTForCausalLM
import torch

# for re-ranking MS-Marco
from transformers import AutoTokenizer, AutoModelForSequenceClassification

########################
####  CHANGE ME 😁  ####
########################
USER_QUESTION = 'What are the inputs and outputs of a Gray code counter?'
NUM_ANSWERS_GENERATED = 5
MAX_TEXT_LENGTH = 512

class TA_Pipeline():
  def __init__(self):
    
    # init to reasonable defaults (these will typically be overwritten when invoked)
    self.user_question = USER_QUESTION
    self.num_answers_generated = NUM_ANSWERS_GENERATED
    self.max_text_length = MAX_TEXT_LENGTH
    
    # DocQuery properties (high RAM)
    self.pipeline = None
    self.doc = None
    
  def contriever(self, user_question: str = USER_QUESTION, num_answers_generated: int = NUM_ANSWERS_GENERATED):
    ''' Invoke contriever (with reasonable defaults).add()
    It finds relevant textbook passages for a given question.
    This can be used for prompting a generative model to generate an better/grounded answer.
    '''
    self.user_question = user_question
    self.num_answers_generated = num_answers_generated
    
    print("User question: ", USER_QUESTION)
    my_contriever = contriever.contriever_final.ContrieverCB()
    contriever_contexts = my_contriever.retrieve_topk(USER_QUESTION, path_to_json = "../data-generator/split_textbook/paragraphs.json", k = num_answers_generated)
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
    
    # print results
    # print('\n\n'.join(top_context_list))
    return top_context_list
  
  def OPT(self, top_context_list):
    """ Run OPT """
    self._load_opt()
    
    response_list = []
    assert NUM_ANSWERS_GENERATED == len(top_context_list)
    for i in range(NUM_ANSWERS_GENERATED):
      prompt = "Please answer this person's question accurately, clearly and concicely. Context: " + top_context_list[i] + '\n' + "Question: " + USER_QUESTION + '\n' + "Answer: "
      inputs = opt_tokenizer(prompt, return_tensors="pt").to("cuda")
      
      generate_ids = opt_model.generate(inputs.input_ids, max_length=MAX_TEXT_LENGTH, do_sample=True, top_k=50, top_p=0.95, temperature=0.95, num_return_sequences=1, repetition_penalty=1.2, length_penalty=1.2, pad_token_id=opt_tokenizer.eos_token_id)
      response = opt_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
      opt_answer = response.split("Answer:")[1]
      response_list.append(opt_answer)
    
    print("Generated Answers:")
    print('\n---------------------------------NEXT---------------------------------\n'.join(response_list))
    return response_list
  
  def _load_opt(self):
    """ Load OPT model """
    global opt_model
    global opt_tokenizer

    opt_model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b").to('cuda') # or use opt-350m
    opt_tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")

  def re_ranking_ms_marco(self, response_list):
    self._load_reranking_ms_marco()
    assert len([USER_QUESTION] * NUM_ANSWERS_GENERATED ) == len(response_list)

    features = rerank_msmarco_tokenizer([USER_QUESTION] * NUM_ANSWERS_GENERATED, response_list,  padding=True, truncation=True, return_tensors="pt")

    rerank_msmarco_model.eval()
    with torch.no_grad():
        scores = rerank_msmarco_model(**features).logits
        print("Scores for each answer (from ms_marco):", scores)
    return scores
  
  def _load_reranking_ms_marco(self):
    global rerank_msmarco_model
    global rerank_msmarco_tokenizer

    rerank_msmarco_model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    rerank_msmarco_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
  def doc_query(self, user_question:str = USER_QUESTION, num_answers_generated: int = 1):
    """ Run DocQuery. Lots of extra dependeicies. 
    TODO: make it so we can save the 'self.doc' object to disk and load it later.
    """
    if not self.pipeline:
      # load docquery on first use
      self._load_doc_query()    
    answer = self.pipeline(question=user_question, **self.doc.context, top_k=num_answers_generated)
    # todo: this has page numbers, that's nice. 
    return answer[0]['answer']
    
  def _load_doc_query(self):
    from docquery import document, pipeline
    self.pipeline = pipeline('document-question-answering')
    # self.doc = document.load_document("../data-generator/notes/Student_Notes_short.pdf") # faster runtime
    self.doc = document.load_document("../data-generator/notes/Student_Notes.pdf")
      
      
class DocQuery():
  """ Class is DEPRICATED """
  def __init__(self):
    pass
  
  def doc_query_single_question(self,user_question, top_k = 1):
    answer = self.pipeline(question=user_question, **self.doc.context, top_k=top_k)
    return answer[0]['answer']
  

  def query(self, data, top_k: int = 5):
    """
    Example return from query
    [
      {
        "questions": " How does the design process for a digital FSM work?",
        "answers": [
            {
                "score": 7.600560638820753e-05,
                "answer": "we represent all types of information using binary digits, or bits.",
                "word_ids": [
                    72,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82
                ],
                "page": 2
            },
            {
                "score": 3.47228669852484e-05,
                "answer": "using binary digits, or bits.",
                "word_ids": [
                    78,
                    79,
                    80,
                    81,
                    82
                ],
                "page": 2
            },
            {
                "score": 2.1325537090888247e-05,
                "answer": "using N-bit 2’s complement,",
                "word_ids": [
                    37,
                    38,
                    39,
                    40
                ],
                "page": 9
            }
        ]
    },    
    """
    from docquery import document, pipeline
    import json
    import re
    import poppler
    
    # Call the DocQuery class
    d = DocQuery()
    with open('../../data-generator/gpt-3/GPT-3_section_level.json') as f:  # use the questions in section data as query
      query = json.load(f)
    # this fn: all_data = d.query(query, 3)
    data = query
    
    # TODO: rework this code to 
    # input: question (maybe list of contexts?)
    # output: answer
    
    
    pipeline = pipeline('document-question-answering')
    doc = document.load_document("../../data-generator/notes/Student_Notes_short.pdf")
    answer = pipeline(question=q, **doc.context, top_k=top_k)
    # doc = document.load_document("../Student_Notes.pdf")
    questions = []
    for i in range(len(data)):
      question = re.sub('\nQ.', '', data[i]['questions'])
      questions.append(question)

    all_data = []
    for q in questions:
    # print(q, p(question=q, **doc.context))
      answer = pipeline(question=q, **doc.context, top_k=top_k)
      data = {}
      data['questions'] = q
      data['answers'] = answer
      all_data.append(data)
      
    
    
    return all_data
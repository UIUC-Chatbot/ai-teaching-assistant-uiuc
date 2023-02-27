import datetime
import json
import os
import time
# from googleapiclient.discovery import build
from typing import List

import openai
from openai.error import RateLimitError


class Prompt_LLMs():

  def __init__(self):
    pass

  def GPT3_response_API(self, prompt: str, max_len: int = 1000) -> str:
    try:
      response = openai.Completion.create(prompt=prompt, model='text-davinci-003', temperature=0.6, max_tokens=max_len, frequency_penalty=1)
    except openai.error.APIConnectionError:
      print("Failed")
    return response['choices'][0]['text']

  def prepare_prompt(self, question: str, context: str, equation: bool = False, cot: bool = False) -> str:
    """prepares prompt based on type of question - factoid, causal, summary, elongate or listing.
        To add equations, use equation = True. To add chain-of-thought, use cot = True"""

    factoid = ["What", "Where"]
    elongate = ["Detail", "Explain", "Discuss", "Expand", "Clarify", "Outline"]
    causal = ["Why", "How"]
    listing = ["List", "Break down"]
    summarize = ["Summarize", "Summarise", "Sum up"]

    first_word = question.split()[0]

    if first_word in factoid:
      prompt = """Generate an objective and logical answer to this question, based on the context. The answer should be short, to-the-point while being substantial as per a freshmen-level language. 
            Do not include any irrelevant information. Give examples."""
    elif first_word in causal:
      prompt = """Generate a reasoning-based, precise answer to this question, based on the context.
            The answer should have a freshmen-level tone and be concise, logic-oriented. 
            Give examples."""
    elif first_word in listing:
      prompt = """Generate a list-type answer to this question, based on the context. 
            The answer should have a freshmen-level tone and be concise. 
            It should contain reasons and examples. """
    elif first_word in elongate:
      prompt = """Generate a detailed, explanatory answer to this question, based on the context. The answer should have a freshmen-level language.  Give examples and talk about real-world applications of the concept. 
            The answer should be long and discuss the concept."""
    elif first_word in summarize:
      prompt = """Summarize this context and answer the question. The answer should have a freshmen-level tone and be concise. 
            Build an in-depth summary using examples."""
    else:
      prompt = """Generate a concise, short and to-the-point answer to this question, based on the context.
            The answer should have a freshmen level easy to understand language and tone. """

    if equation == True:
      prompt = prompt + "Add all necessary equations to explain this."
    elif cot == True:
      prompt = "Generate a short answer to this question, based on the context. Use freshmen-level language." + "Let's think step by step."
    else:
      prompt = prompt

    fin_prompt = prompt + "\nContext" + context.replace("\n", " ") + "\nQuestion:" + question.replace("\n", " ") + "\nAnswer:"

    return fin_prompt

  def GPT3_fewshot(self, question: str) -> str:
    """used in TA_gradio_ux.py"""

    examples = """ 
        Question : What is De Morgan's Law?
        Instruction : Generate an objective and logical answer to this question, based on the context. The answer should be short, to-the-point while being substantial as per a freshmen-level language.  Do not include any irrelevant information. Give examples. Add all the necessary equations. 
        Answer: De Morgan's Law is a way to find out an alternative representation of a given boolean logic. Given two variables A and B, the two main forms to remember are:(A + B)' = A' B' (The NOT of A OR B is equal to NOT A AND NOT B)
        (AB) ' = A' + B' (The NOT of A AND B is equal to NOT A OR NOT B)

        Question : How do I check for overflow in a 2's complement operation?
        Instruction : Generate a concise and precise answer to this question. Give a real-world example and specify how overflow is indicated. Answer using freshmen-level language. Do not include any irrelevant information.
        Answer : Overflow can be indicated in a 2's complement if the result has the wrong sign, such as if 2 positive numbers sum to a negative number or if 2 negative numbers sum to positive numbers. 

        Question : Why would I use fixed-point representation?
        Instruction : Generate a concise answer to this question specifying the particular use-case when fixed point representation should be used. Answer using freshmen-level language. Do not include any irrelevant information.
        Answer : Fixed-point operations can be implemented with integer arithmetic, which is typically faster than floating-point arithmetic.

        Question : What is the difference between clock synchronous and clock asynchronous designs?
        Instruction : Generate a list-type, precise and short answer to this question, clearly stating the differences between the two. Answer using freshmen-level language. Do not include any irrelevant information.
        Answer : Clock synchronous designs use a system clock to control the timing of all circuit operations. The circuit elements are updated on either the rising or falling edge of the clock signal. Clock asynchronous designs do not rely on a system clock to synchronize the timing of circuits, the circuit elements are designed to operate independently of the clock, using their own timing control.  
        """

    new_question = examples + "Question : " + question
    response = self.GPT3_response_API(new_question)  #few shot
    resp_answer = response.split('\n')[-1]  #get answer
    #instruction = response.split('\n')[-2] #get the gpt-3 generated prompt

    return resp_answer

  def google_links(self, question: str) -> List[str]:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    SEARCH_ENGINE_ID = os.environ.get('SEARCH_ENGINE_ID')
    query = question
    service = build('customsearch', 'v1', developerKey=GOOGLE_API_KEY)
    # Search for web pages using the Google Custom Search API
    result = service.cse().list(q=query, cx=SEARCH_ENGINE_ID).execute()
    links = []
    # Get the most relevant web page links
    for item in result['items']:
      links.append(item['link'])

    return links

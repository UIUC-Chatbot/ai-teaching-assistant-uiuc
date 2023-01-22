import openai
import time
import json
import os
import backoff
from openai.error import RateLimitError
import datetime
import math

openai.api_key=os.getenv("OPENAI_API_KEY")

"""GPT-3 MODEL API"""

class GPT3_model: 
    def __init__(self):
        super(GPT3_model,self).__init__()
    def prepare_prompt(self, context:str,question:str) -> str :
        factoid = ["What", "Where", "When", "Explain", "Discuss", "Clarify"]
        causal = ["Why", "How"]
        listing = ["List", "Break down"]
        if any(word in question for word in factoid):
            prompt = """Generate an objective, formal and logically sound answer to this question, based on the given context. 
            The answer must spur curiosity, enable interactive discussions and make the user ask further questions. 
            It should be interesting and use advanced vocabulary and complex sentence structures.
            Context : """ + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"
        elif any(word in question for word in causal):
            prompt = """Generate a procedural, knowledgeable and reasoning-based answer about this question, based on the given context. 
            The answer must use inference mechanisms and logic to subjectively discuss the topic. It should be creative and logic-oriented, analytical and extensive. Context :""" + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"
        elif any(word in question for word in listing):
            prompt = """Generate a list-type, descriptive answer to this question, based on the given context. 
            The answer should be very detailed and contain reasons, explanations and elaborations about the topic. It should be interesting and use advanced vocabulary and complex sentence structures. Context :""" + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"
        else:
            prompt = """Generate a detailed, interesting answer to this question, based on the given context. 
            The answer must be engaging and provoke interactions. It should use academic language and a formal tone. 
            Context : """ + context.replace("\n"," ") + "\nQuestion:" + question.replace("\n"," ") + "\nAnswer:"
        return prompt
    @backoff.on_exception(backoff.expo, RateLimitError)
    def GPT3_response_API(self, prompt:str, max_len:int=300) -> str :
        try:
            response = openai.Completion.create(prompt=prompt,
            model= 'text-davinci-003',
            temperature = 0.6,
            max_tokens = max_len,
            best_of = 5,
            n = 3,
            frequency_penalty = 1)
        except openai.error.APIConnectionError:
            print("Failed")    
        return response['choices'][0]['text']
    def calculate_gpt3_cost(self, text):
        """ 
        returns total tokens and total price. Note - function doesn't take into account best_of and n.
        pricing - the formula used to perform this calculation is from: https://beta.openai.com/pricing
        """ 
        total_tokens = math.ceil((len(text) - text.count(' ')) / 4)
        return total_tokens, total_tokens * 0.0200/1000
    def gpt3_answer_question(self, context:str, question:str, max_len:int=300):
        prompt = self.prepare_prompt(context,question)
        return self.GPT3_response_API(prompt) #change args here if req.

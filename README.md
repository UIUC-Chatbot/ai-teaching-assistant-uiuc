# "Better than Google" Multi-Media QA & Search for Electrical Engineering at UIUC

## System diagram
Our system runs 11 separate models in parallel for text/image retrieval, generation, moderation and ranking and still achieves a median 2-second response time.

<img width="4083" alt="NCSA AI Teaching Assistant -- Detailed diagram (5)" src="https://user-images.githubusercontent.com/13607221/226132728-0f50bd25-50d9-4e5e-b162-ae6c8ca47f33.png">

## Data 
We use data from textbooks, lecture videos, and student QA forums (ordered, subjuctively, by importance). None of this data is currently available publically because this project was not granted those rights by the authors.

### RLHF 
My favorite contribution is the novel approach of semantic search retrieval during RLHF, using a dataset I iteratively produced by hiring a team of five Electrical Engineering students. That data is freely available on Huggingface here: https://huggingface.co/datasets/kastan/rlhf-qa-comparisons. We specifically cover the material necessary in the UIUC course ECE 120, intro to Electrical Engineering.

# Evaluation
We have a lot of models here. To evaluate which ones are helping, and which hurt, every time we push a new feature we re-run this evaluation. Our evaluation dataset of QA pairs is produced in house, written by expert electrical engineers. Using these questions, we generate answers with each of our models. Finally, we ask GPT-3 if the generated are "better" or "worse" than the ground truth answers written by humans. One limitation is that GPT-3 evaluates itself. GPT-3 nearly always thinks that GPT-3 is great, which is probably not true and a limitation of this evaluation method. Maybe we should run this same evaluation with Cohere's models to compare.

Nevertheless, iterative evaluation was crucial to ensure our new features were making our system better.

See the [full evalaution results here](https://github.com/UIUC-Chatbot/ai-teaching-assistant-uiuc/blob/main/Evaluation_Results.pdf). See the [evaluation code here](https://github.com/UIUC-Chatbot/ai-teaching-assistant-uiuc/blob/main/evaluation.py).

<img width="837" alt="Bar chart showing GPT-3 is the best, with ChatGPT in 2nd place and OpenAssistant in 3rd place." src="https://user-images.githubusercontent.com/13607221/228375233-4f27f85d-10bc-4383-9eb7-41445dc71638.png">


# Usage
This project is fully open source, with the exception of commercial textbooks. **I highly encourage you to simply plug in your own [Pinecone database](https://www.pinecone.io/) of documents and use this in your work!**

## 🚀 Install

1. Python requirements
```bash
# Tested on python 3.8
pip install -r requirements.txt
```

2. API keys

We rely on these APIs enumerated in `run_ta_gradio.sh`. Be sure to add your own 😄

3. Document store for retrieval-augmented generation 

Simply build your own Pinecone database of your documents. We open source the scripts we use for this, where our source material is in PDF or plaintext and our lecture slides are .jpg images (sneakily exported from .pptx). 

Data cleaning utils:
* [Textbook PDF to Pinecone database](https://github.com/UIUC-Chatbot/data-generator/blob/main/textbooks_to_vectorDB/data_formatting_patel.ipynb)
* [Video transcripts (from Whisper) to Pinecone](https://github.com/UIUC-Chatbot/data-generator/blob/main/textbooks_to_vectorDB/data_formatting_whisper_transcripts_Vlad_lectures.ipynb)

And you're done! You can now run the app 🎉 

```bash
# run web app!
bash run_ta_gradio.sh
```

## Files
- `main.py`: The main aggregator of our many LLM implementations.
- `TA_gradio_ux.py`: Defines the Gradio UX, and calls models defined in `main.py`.
- `prompting.py`: Our home for prompt engineering.
- `evaluation.py`: Run GPT-3 powered evaluation of our system. We routinely run this to determine if each additional feature makes responses "better" or "worse".
- `feedback.json`: Collection of real user feedback via the Gradio web app.
- `run_ta_gradio.sh`: Entrypoint launch script.

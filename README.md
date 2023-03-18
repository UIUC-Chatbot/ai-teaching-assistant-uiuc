# "Better than Google" Multi-Media QA & Search for Electrical Engineering at UIUC

## System diagram
Our system runs 11 separate models in parallel for text/image retrieval, generation, moderation and ranking and still achieves a median 2-second response time.

<img width="4083" alt="NCSA AI Teaching Assistant -- Detailed diagram (5)" src="https://user-images.githubusercontent.com/13607221/226132728-0f50bd25-50d9-4e5e-b162-ae6c8ca47f33.png">

## Data 
We use data from textbooks, lecture videos, and student QA forums (ordered, subjuctively, by importance). None of this data is currently available publically because this project was not granted those rights by the authors.

### RLHF 
my favorite contribution is the novel approach of semantic search retrieval during RLHF, using a dataset I iteratively produced by hiring a team of five Electrical Engineering students. That data is freely available on Huggingface here: https://huggingface.co/datasets/kastan/rlhf-qa-comparisons. We specifically cover the material necessary in the UIUC course ECE 120, intro to Electrical Engineering.

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

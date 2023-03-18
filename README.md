# "Better than Google" Multi-Media QA & Search for Electrical Engineering at UIUC

## System diagram
Our system runs 11 separate models in parallel for text/image retrieval, generation, moderation and ranking and still achieves a median 2-second response time.

<img width="4083" alt="NCSA AI Teaching Assistant -- Detailed diagram (5)" src="https://user-images.githubusercontent.com/13607221/226132728-0f50bd25-50d9-4e5e-b162-ae6c8ca47f33.png">

## Data 
We use data from textbooks, lecture videos, and student QA forums (ordered, subjuctively, by importance). None of this data is currently available publically because this project was not granted those rights by the authors.

### RLHF 
my favorite contribution is the novel approach of semantic search retrieval during RLHF, using a dataset I iteratively produced by hiring a team of five Electrical Engineering students. That data is freely available on Huggingface here: https://huggingface.co/datasets/kastan/rlhf-qa-comparisons. We specifically cover the material necessary in the UIUC course ECE 120, intro to Electrical Engineering.

## Files
- `main.py`: A main aggregator of our many LLM implementations.
- `TA_gradio_ux.py`: Defines the Gradio UX, and calls models defined in `main.py`.
- `prompting.py`: Our home for prompt engineering.
- `run_ta_gradio.sh`: Entrypoint launch script.

export OPENAI_API_KEY=<YOUR_KEY_HERE>
export PINECONE_API_KEY=<YOUR_KEY_HERE>
export GOOGLE_API_KEY=<YOUR_KEY_HERE>
export SEARCH_ENGINE_ID=<YOUR_KEY_HERE>
# use shared cache to reduce model duplication
export TRANSFORMERS_CACHE=/mnt/project/chatbotai/huggingface_cache/transformers
export HF_DATASETS_CACHE=/mnt/project/chatbotai/huggingface_cache/datasets

python TA_gradio_ux.py

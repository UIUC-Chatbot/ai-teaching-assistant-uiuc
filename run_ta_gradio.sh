export OPENAI_API_KEY=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
python TA_gradio_ux.py \
    --model_weight "/home/wentaoy4/lgm/data/model_weight/opt_finetune_b128_e20_lr5e06.pt" \
    --device "cuda:0" \
    --wandb_entity "wentaoy" \
    --wandb_project "TA_Chatbot" \
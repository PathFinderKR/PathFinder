"""
lm_eval --model hf \
    --model_args pretrained=/home/pathfinder/models/PathFinderKR/GPT2-small-2025-10-25_10-12-10,trust_remote_code=True,tokenizer=gpt2 \
    --tasks arc_easy,lambada_openai,hellaswag,winogrande \
    --num_fewshot 5 \
    --device cuda:0 \
    --batch_size 16 \
    --seed 1234 \
    --output_path results \
    --log_samples \
    --wandb_args project=lm-eval-harness-integration
"""
MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
MAX_LENGTH=4096
PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --tasks=arc_challenge,ifeval,hellaswag
PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --num_fewshot=8 --tasks=gsm8k

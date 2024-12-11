# MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="Llama-3_2-1B-Instruct"
# EXP_DIR="Llama_wikitext2_lr1e-04_fqlr1e-05_wd1e-04_n1024_bs32"

MODEL_ID="HuggingFaceTB/SmolLM-1.7B-Instruct"
MODEL_NAME="SmolLM-1_7B-Instruct"
EXP_DIR="SmolL_wikitext2_lr1e-04_fqlr1e-05_wd1e-04_n1024_bs32"

INIT_DIR="FQ_4bit_no_embed_svd_rank256_g64_hybrid_rand_quant100+_sqrtS"
MAX_LENGTH=4096

NNCF_CKPT_DIR=/local_ssd2/nlyalyus/MODEL_DIR/$MODEL_NAME/$INIT_DIR/
PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH,nncf_ckpt_dir=$NNCF_CKPT_DIR --tasks=arc_challenge
PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH,nncf_ckpt_dir=$NNCF_CKPT_DIR --tasks=hellaswag
PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH,nncf_ckpt_dir=$NNCF_CKPT_DIR --tasks=ifeval

# arc_challenge
# hellaswag
# ifeval
# PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH,nncf_ckpt_dir=$NNCF_CKPT_DIR --tasks=squadv2 --num_fewshot=1
# PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH,nncf_ckpt_dir=$NNCF_CKPT_DIR --tasks=gsm8k --num_fewshot=8
# PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH,nncf_ckpt_dir=$NNCF_CKPT_DIR --tasks=drop --num_fewshot=3


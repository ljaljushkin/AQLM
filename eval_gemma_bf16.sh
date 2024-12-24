
MODEL_DIR="/local_ssd2/nlyalyus/MODEL_DIR"
MODEL_ID="google/gemma-2-2b-it"
MODEL_NAME="gemma-2-2b-it"

MAX_LENGTH=4096

NNCF_CKPT_DIR=$MODEL_DIR/$MODEL_NAME

# TASK="gsm8k"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# CUDA_VISIBLE_DEVICES=7 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK --num_fewshot=8 > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

# TASK="ifeval"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=6 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"

TASK="arc_challenge"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=7 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"

TASK="hellaswag"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=4 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"

# --tasks=squadv2 --num_fewshot=1
# --tasks=drop --num_fewshot=3


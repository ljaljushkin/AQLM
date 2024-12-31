MODEL_DIR="/local_ssd2/nlyalyus/MODEL_DIR"
MODEL_ID="HuggingFaceTB/SmolLM-1.7B-Instruct"
MODEL_NAME="SmolLM-1_7B-Instruct"

INIT_DIR="FQ_emb_head_int8_sym_int4_sym_rank256_gs-1_real_ss"
EXP_DIR="SmolL_lr5e-04_fqlr5e-05_wd5e-04_tune_both_signed"

MODEL_DIR="/local_ssd2/nlyalyus/MODEL_DIR"
MAX_LENGTH=2048
NNCF_CKPT_DIR=$MODEL_DIR/$MODEL_NAME/$INIT_DIR/$EXP_DIR


TASK="arc_challenge"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"

TASK="hellaswag"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=2 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"

TASK="mmlu"
LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
CUDA_VISIBLE_DEVICES=3 PYTHONIOENCODING=utf-8 lm_eval --model=hf --model_args=pretrained=$MODEL_ID,nncf_ckpt_dir=$NNCF_CKPT_DIR,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH --output_path=$OUTPUT_PATH --tasks=$TASK --num_fewshot 5 --batch_size 4 > $LOG_FILE 2>&1  &
pid=$!
echo "The process ID: $pid, log file: $LOG_FILE"

# cd ../nncf
# TASK="WWB"
# LOG_FILE=$NNCF_CKPT_DIR/"${MODEL_NAME}_${TASK}.log"
# OUTPUT_PATH=$NNCF_CKPT_DIR/"results_$TASK.json"
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python wwb_eval.py -m=$MODEL_ID -n=$NNCF_CKPT_DIR > $LOG_FILE 2>&1  &
# pid=$!
# echo "The process ID: $pid, log file: $LOG_FILE"
# cd -

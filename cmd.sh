#!/bin/bash

set -e

# run_commands() {
#     local directory=$1
#     shift
#     local rank_names=("$@")

#     cd $ROOT_DIR
#     cd "$directory" || exit
#     for nncf_ckpt_dir in "${rank_names[@]}"
#     do
#         export nncf_ckpt_dir
#         command=$(echo $eval_command_template | envsubst)
#         echo "Running: $command"
#         eval $command 2>&1 | tee -a 'eval.log'
#     done
# }
# --base_model=HuggingFaceTB/SmolLM-1.7B-Instruct \
# --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/SmolLM-1_7B-Instruct/FQ_4bit_no_embed_svd_rank\${rank}_g64_hybrid_rand_quant100+_sqrtS/ \
# --base_model=microsoft/Phi-3.5-mini-instruct \

tune_command_template="PYTHONIOENCODING=utf-8 python finetune.py \
--nncf_ckpt_dir=$HOME/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd_rank\${rank}_g64_hybrid_rand_quant100+_sqrtS/ \
--base_model=microsoft/Phi-3-mini-4k-instruct \
--model_seqlen=\$model_seqlen \
--val_size=0   \
--adam_beta1=0.90  \
--adam_beta2=0.999  \
--early_stop=3 \
--batch_size=\$batch_size \
--microbatch_size=\$microbatch_size \
--trust_remote_code  \
--keep_best_model \
--device_map=auto \
--nsamples=\$nsamples \
--weight_decay=\$weight_decay \
--dataset=\$dataset \
--lr=\$lr \
--fq_lr=\${fq_lr} \
--num_blocks=\$num_blocks \
--frequency=\$frequency \
--lr_scale=\$lr_scale \
--warmup=\$warmup \
--dtype=bfloat16 \
--finetune_dtype=bfloat16 \
--mlflow"
# --qloss \
# --exp_name=slm_const_lr2e-04_fqlr1e-03_wd1e-03_rand100+_qloss_n1024_r1"
# --print_every_steps=1"
# --print_every_steps=1 \
# --exp_name=debug \
# Why is the convergence better for FQ with finetune_dtype=float32 and bfloat16 weights
# --amp"
# ALPACA
# "--wandb_project=trainer_tune"
# DISTILLATION

weight_decays=1e-4 #2e-4 1e-2) #(0 1e-5 1e-2)
rank=256
model_seqlen=1024
batch_sizes=32 #(128 64) #32
microbatch_size=2 #2 #2
list_nsamples=1024 #128 #128
dataset=wikitext2
lrs=1e-4
fq_lrs=1e-5
lr_scale=0 # 1 2)
num_blocks=32 # 8)
frequencys=32 #2 #(8 16 32)
warmup=0 #(6 16 32)

for batch_size in "${batch_sizes[@]}"
do
    for lr in "${lrs[@]}"
    do
        for weight_decay in "${weight_decays[@]}"
        do
            for fq_lr in "${fq_lrs[@]}"
            do
                for nsamples in "${list_nsamples[@]}"
                do
                    for frequency in "${frequencys[@]}"
                    do
                        export rank model_seqlen batch_size microbatch_size nsamples weight_decay dataset lr fq_lr lr_scale num_blocks frequency warmup
                        command=$(echo $tune_command_template | envsubst)
                        echo "Running: $command"
                        eval $command 2>&1 | tee -a tune_$(date '+%Y-%m-%d %H:%M:%S').log

                        # run_commands "FQ_4bit_no_embed_svd_rank8_g64" "${ckpt_dir[@]}"
                    done
                done
            done
        done
    done
done

# eval_command_template='lm_eval --model=hf --model_args=pretrained=microsoft/Phi-3-mini-4k-instruct,trust_remote_code=True,nncf_ckpt_dir=$nncf_ckpt_dir --tasks=wikitext'
# overfit experiments
# command_template='python finetune.py --base_model=microsoft/Phi-3-mini-4k-instruct --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_31layer_svd_debug --model_seqlen=$model_seqlen --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=$batch_size  --microbatch_size=$microbatch_size --trust_remote_code  --keep_best_model --nsamples=$nsamples --dtype=auto --weight_decay=$weight_decay --device_map=auto --amp --dataset=$dataset --lr=$lr --num_blocks=$num_blocks --frequency=$frequency --lr_scale=$lr_scale --epochs 5 --exp_name debug --print_every_steps=1'


# seq_len 4096
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_31layer_svd/ --model_seqlen=4096 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --keep_best_model --nsamples=128 --skip_first_eval --dtype=auto --device_map=auto --amp --dataset=wikitext2 --wandb --lr=1e-4 --epochs 5


#############################################################################    WWB    ###################################
# pip install whowhatbench@git+https://github.com/andreyanufr/openvino.genai.git@837294cb21a9bb408faa346ddde287ea748ee22c#subdirectory=tools/who_what_benchmark
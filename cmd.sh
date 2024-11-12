#!/bin/bash

# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=5e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=4 --frequency=6
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=5e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=4 --frequency=2
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=5e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=2 --frequency=2
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=1e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=2 --frequency=2
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=5e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=8 --frequency=8
# set -e
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=5e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=4 --frequency=2 --lr_scale=10
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=5e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=4 --frequency=2 --lr_scale=5
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=5e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=8 --frequency=1 --lr_scale=1

# not finished
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=5e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=8 --frequency=1 --lr_scale=5
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd/  --dataset=pajama  --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --eval_every_steps=64  --keep_best_model --nsamples=128 --lr=5e-4 --skip_first_eval --dtype=auto --device_map=auto --amp --wandb --num_blocks=8 --frequency=8 --lr_scale=5

# 31 layer
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_31layer_svd/ --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --keep_best_model --nsamples=128 --skip_first_eval --dtype=auto --device_map=auto --amp --dataset=wikitext2 --wandb --lr=1e-4 --epochs 5
# --weight_decay=1e-5

# cosine lr=3e-4
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_31layer_svd/ --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --keep_best_model --nsamples=128 --skip_first_eval --dtype=auto --device_map=auto --amp --dataset=wikitext2 --wandb --lr=3e-4 --epochs 5

# cosine lr=3e-4 warmup for 1/3 epoch
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_31layer_svd/ --model_seqlen=1024 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --keep_best_model --nsamples=128 --skip_first_eval --dtype=auto --device_map=auto --amp --dataset=wikitext2 --wandb --lr=3e-4 --epochs 5 --warmup

# const lr=1e-4, FQ lr=1e-5
# const lr=1e-5, FQ lr=1e-4

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

tune_command_template="python finetune.py \
--base_model=microsoft/Phi-3-mini-4k-instruct \
--nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd_rank\${rank}_g64_bfloat16/ \
--model_seqlen=\$model_seqlen \
--val_size=0   \
--adam_beta1=0.90  \
--adam_beta2=0.999  \
--early_stop=3 \
--batch_size=\$batch_size \
--microbatch_size=\$microbatch_size \
--trust_remote_code  \
--keep_best_model \
--nsamples=\$nsamples \
--dtype=bfloat16 \
--load_dtype=bfloat16 \
--finetune_dtype=bfloat16 \
--amp_dtype=bfloat16 \
--weight_decay=\$weight_decay \
--device_map=auto \
--dataset=\$dataset \
--lr=\$lr \
--fq_lr=\${fq_lr} \
--num_blocks=\$num_blocks \
--frequency=\$frequency \
--lr_scale=\$lr_scale \
--warmup=\$warmup"
# --wandb
# --amp

weight_decays=1e-4 #(0 1e-5 1e-2)
rank=256
model_seqlen=1024
batch_size=32
microbatch_size=2 #2
list_nsamples=1028 #128 #128
dataset=wikitext2
lrs=1e-4 #(5e-4 1e-4)
fq_lrs=1e-5 #1e-3)
lr_scale=0 # 1 2)
num_blockss=32 # 8)
frequencys=16 #)
warmup=0

for num_blocks in "${num_blockss[@]}"
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
                        eval $command 2>&1 | tee -a 'tune.log'

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




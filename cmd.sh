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
command_template='python finetune.py --base_model=microsoft/Phi-3-mini-4k-instruct --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_no_embed_svd_rank${rank} --model_seqlen=$model_seqlen --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=$batch_size  --microbatch_size=$microbatch_size --trust_remote_code  --keep_best_model --nsamples=$nsamples --dtype=auto --weight_decay=$weight_decay --device_map=auto --amp --dataset=$dataset --lr=$lr --num_blocks=$num_blocks --frequency=$frequency --lr_scale=$lr_scale --wandb'

weight_decay=0
ranks=8
model_seqlen=1024
batch_size=32
microbatch_size=2
list_nsamples=512
dataset=wikitext2
lr=1
lr_scale_values=0 # 1 2)
num_blocks=4
frequency=4

# for model_seqlen in "${seqlens[@]}"
# do
for lr_scale in "${lr_scale_values[@]}"
do
    for nsamples in "${list_nsamples[@]}"
    do
        for rank in "${ranks[@]}"
        do
            export rank model_seqlen batch_size microbatch_size nsamples weight_decay dataset lr lr_scale num_blocks frequency
            command=$(echo $command_template | envsubst)
            echo "Running: $command"
            eval $command
        done
    done
done
# done

# seq_len 4096
# python finetune.py  --base_model=microsoft/Phi-3-mini-4k-instruct  --nncf_ckpt_dir=/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct/FQ_4bit_31layer_svd/ --model_seqlen=4096 --val_size=0   --adam_beta1=0.90  --adam_beta2=0.999  --early_stop=3 --batch_size=4  --microbatch_size=2 --trust_remote_code  --keep_best_model --nsamples=128 --skip_first_eval --dtype=auto --device_map=auto --amp --dataset=wikitext2 --wandb --lr=1e-4 --epochs 5




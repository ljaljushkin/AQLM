# ROOT_DIR="/home/nlyaly/MODEL_DIR/Phi-3_5-mini-instruct"
# command_template='lm_eval --model=hf --model_args=pretrained=microsoft/Phi-3.5-mini-instruct,trust_remote_code=True,nncf_ckpt_dir=${nncf_ckpt_dir},dtype=auto,device_map=auto,parallelize=True,max_length=4096 --tasks=wikitext'

# ROOT_DIR="/home/nlyaly/MODEL_DIR/SmolLM-1_7B-Instruct"
# command_template='lm_eval --model=hf --model_args=pretrained=HuggingFaceTB/SmolLM-1.7B-Instruct,trust_remote_code=True,nncf_ckpt_dir=${nncf_ckpt_dir},device_map=auto,parallelize=True,dtype=bfloat16 --tasks=wikitext'

ROOT_DIR="/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct"
command_template='lm_eval --model=hf --model_args=pretrained=microsoft/Phi-3-mini-4k-instruct,trust_remote_code=True,nncf_ckpt_dir=${nncf_ckpt_dir},device_map=auto,parallelize=True,dtype=bfloat16 --tasks=wikitext'

rank8_names=(
    # weekend_rank8_g64_seqlen1024_lr0.0005_lr_scale0_n128
    # weekend_rank8_seqlen1024_lr0.0001_lr_scale0
    # weekend_rank8_seqlen1024_lr0.0001_lr_scale5
    # tune8_epoch8_lr0.0005
    # tune2_epoch2_lr0.0005
    # tune_both_g64_rank8_lr1e-04_wd0e+00_n128_fqlr1e-04_freq8
    # tune_both_g64_rank8_lr5e-04_wd1e-02_n128_fqlr1e-04_freq2
    # tune_both_g64_rank8_lr5e-04_wd0e+00_n128_fqlr1e-03_freq2
    # tune_both_g64_rank8_lr1e-04_wd0e+00_n128_fqlr1e-04_freq2
    # tune_fq_g64_rank256_lr0e+00_wd0e+00_n1024_fqlr1e-05_freq4_warm0
    # tune_both_after_fq_g64_rank256_lr1e-05_wd1e-04_n1028_fqlr1e-06_freq2
    # tune_both_g64_rank256_lr1e-04_wd1e-04_n1028_fqlr1e-05_freq32 # 10.10 word_ppl
    # tune_both_g64_rank256_lr5e-04_wd5e-04_n1028_fqlr5e-05_freq32
    # 3.5_cosine_g64_rank256_lr2e-04_wd0e+00_n1024_fqlr1e-04_freq4_warm0 # 11.15
    # 3.5_const_g64_rank256_lr2e-04_wd0e+00_n1024_fqlr1e-04_freq16_warm0  # ??
    # 3.5_cosine_g64_rank256_lr2e-04_wd0e+00_n1024_fqlr1e-04_freq32_warm0 # 10.58
    # 3.5_cosine_g64_rank256_lr2e-04_wd0e+00_n1024_fqlr1e-04_freq16_warm0 # 10.61
    # 3.5_cosine_g64_rank256_lr2e-04_wd0e+00_n1024_fqlr1e-04_freq8_warm0 # 10.74
    # 3.5_cosine_g64_rank256_lr2e-04_wd0e+00_n1024_fqlr1e-04_freq4_warm6 # 10.68
    # 3.5_tune_both_g64_rank256_lr1e-04_wd1e-04_n1024_fqlr1e-05_freq32 # 10.74 rank8
    # slm_const_g64_rank256_lr2e-04_n1024_fqlr1e-04_freq32
    # slm_cosine_g64_rank256_lr2e-04_n1024_fqlr1e-04_freq32
    # slm_const_g64_rank256_lr2e-04_n1024_fqlr1e-04_freq32_rand
    # slm_const_both_g64_rank256_lr1e-04_n1024_fqlr1e-03_wd1e-04_bs32_rand100+_qloss_10xB
    # slm_const_lr2e-04_fqlr1e-03_wd1e-03_rand100+_qloss_n1024_r1
    # slm_const_both_g64_rank256_lr1e-04_n1024_fqlr1e-03_wd0e+00_bs32_rand100+_qloss_10xB
    slm_const_lr1e-04_fqlr1e-03_wd1e-03_rand100+_qloss_n128_r0.5
)

# svd_names=(
#     # FQ_4bit_no_embed_svd_rank8_g64
#     # FQ_4bit_no_embed_svd_rank256_g64_float32
#     FQ_4bit_no_embed
# )
# rank32_names=(
#     weekend_rank32_seqlen1024_lr0.0005_lr_scale0_n128
#     weekend_rank32_seqlen1024_lr0.0005_lr_scale0_n1024
# )

run_commands() {
    local directory=$1
    shift
    local rank_names=("$@")

    cd $ROOT_DIR
    cd "$directory" || exit
    for nncf_ckpt_dir in "${rank_names[@]}"
    do
        export nncf_ckpt_dir
        command=$(echo $command_template | envsubst)
        echo "Running: $command"
        eval $command 2>&1 | tee -a 'eval.log'
    done
}



# run_commands "FQ_4bit_no_embed_svd_rank256_g64_bfloat16" "${rank8_names[@]}"
# run_commands "FQ_4bit_no_embed_svd_rank256_g64_float32" "${rank8_names[@]}"
 run_commands "FQ_4bit_no_embed_svd_rank256_g64_hybrid_rand_quant100+" "${rank8_names[@]}"

# run_commands "./" "${svd_names[@]}"

# run_commands "FQ_4bit_no_embed_svd_rank256_g64/tune_fq_g64_rank256_lr0e+00_wd0e+00_n1024_fqlr1e-05_freq4_warm0/last_ckpt" "${rank8_names[@]}"
# run_commands "FQ_4bit_no_embed_svd_rank32" "${rank32_names[@]}"







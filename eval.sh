ROOT_DIR="/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct"
command_template='lm_eval --model=hf --model_args=pretrained=microsoft/Phi-3-mini-4k-instruct,trust_remote_code=True,nncf_ckpt_dir=${nncf_ckpt_dir}/last_ckpt,dtype=float16 --tasks=wikitext'

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
    tune_both_after_fq_g64_rank256_lr1e-05_wd1e-04_n1028_fqlr1e-06_freq2
)

svd_names=(
    FQ_4bit_no_embed_svd_rank8_g64
)
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




run_commands "FQ_4bit_no_embed_svd_rank256_g64/tune_fq_g64_rank256_lr0e+00_wd0e+00_n1024_fqlr1e-05_freq4_warm0/last_ckpt" "${rank8_names[@]}"
# run_commands "./" "${svd_names[@]}"
# run_commands "FQ_4bit_no_embed_svd_rank32" "${rank32_names[@]}"







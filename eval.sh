ROOT_DIR="/home/nlyaly/MODEL_DIR/Phi-3-mini-4k-instruct"
command_template='lm_eval --model=hf --model_args=pretrained=microsoft/Phi-3-mini-4k-instruct,trust_remote_code=True,nncf_ckpt_dir=$nncf_ckpt_dir --tasks=wikitext'

rank8_names=(
    weekend_rank8_g64_seqlen1024_lr0.0005_lr_scale0_n128
    # weekend_rank8_seqlen1024_lr0.0001_lr_scale0
    # weekend_rank8_seqlen1024_lr0.0001_lr_scale5
    # tune8_epoch8_lr0.0005
    # tune2_epoch2_lr0.0005
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




# run_commands "FQ_4bit_no_embed_svd_rank8_g64" "${rank8_names[@]}"
run_commands "./" "${svd_names[@]}"
# run_commands "FQ_4bit_no_embed_svd_rank32" "${rank32_names[@]}"







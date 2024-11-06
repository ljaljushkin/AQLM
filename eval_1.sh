lm_eval \
--model=hf \
--model_args=\
pretrained=microsoft/Phi-3-mini-4k-instruct,trust_remote_code=True,\
nncf_ckpt_dir=weekend_rank8_seqlen1024_lr0.0001_lr_scale1 \
--tasks=wikitext \
# nncf_ckpt_dir=weekend_rank8_seqlen1024_lr0.0005_lr_scale0,\
# dtype=float16 \


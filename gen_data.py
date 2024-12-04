# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
from tqdm import tqdm
from pathlib import Path

# MODEL_ID = 'google/gemma-2-2b-it'
# MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
# MODEL_ID = 'microsoft/Phi-3.5-mini-instruct'
# MODEL_ID = 'microsoft/Phi-3-mini-4k-instruct'
# MODEL_ID = 'HuggingFaceTB/SmolLM-1.7B-Instruct'
MODEL_ID = 'google/gemma-2-2b-it'

MODEL_NAME = Path(MODEL_ID).name.replace('.', '_')

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print("Tokenizer loaded!")
print("Loading model")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.cuda()
print("Model loaded!")




dataset_size = 700
vocab_size_names = ["padded_vocab_size", "vocab_size"]
vocab_size = tokenizer.vocab_size
for vocab_size_name in vocab_size_names:
    if hasattr(model.config, vocab_size_name):
        vocab_size = getattr(model.config, vocab_size_name)

step_num = max(1, vocab_size // dataset_size)


# i_start = 1#sys.argv[1]
# n_vocab = 128 # number of initial tokens for synthesizing data on each GPU.
# if os.path.exists("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl"):
#     with open("gen_data/gen.chunk."+str(i_start).zfill(2)+".jsonl", "r") as f:
#         lines = f.readlines()
#         inner_loop = len(lines) % n_vocab
#         outer_loop = len(lines) // n_vocab
# else:
inner_loop = 0
outer_loop = 0

if not os.path.exists("gen_data"):
    os.mkdir("gen_data")



th=5
# for j in range(3 + outer_loop, 6):
j=3
# for i in tqdm(range(int(i_start) * n_vocab + inner_loop, (int(i_start)+1) * n_vocab)):
ids_counter = 42
print('dataset_size', dataset_size)
print('tokenizer.vocab_size', tokenizer.vocab_size)
print('vocab_size', vocab_size)
# print('step_num', step_num)
# with tqdm(total=dataset_size, description="Generating text data") as pbar:
pbar = tqdm(total=dataset_size, desc="Generating text data")
num_generated = 0
while num_generated < dataset_size:
    ids_counter = random.randint(0, vocab_size)
    print(ids_counter)
    input_ids = torch.tensor([[ids_counter]]).cuda()
    start_token = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    print(f'Start token id: {input_ids[0]} start token text: {repr(start_token)}')
    # input_text = "Write me a poem about Machine Learning."
    # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    # outputs = model.generate(**input_ids, max_new_tokens=32)

    instruction = 'Continue the text in an autoregressive manner, starting from this word: '
    prompt = instruction + start_token
    # print(f'prompt: {repr(prompt)}')
    messages = [
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
    outputs1 = model.generate(**input_ids, do_sample=False, max_new_tokens=j)
    # outputs1 = model.generate(input_ids, do_sample=False, max_length=j)
    first_j_tids = outputs1[:, input_ids['input_ids'].shape[1]:]
    first_j_tokens = tokenizer.batch_decode(first_j_tids, skip_special_tokens=True)[0]
    # first_j_tokens = first_j_tokens[len(prompt)+5:]
    print(f'Generated first {j} symbol ids: {first_j_tids} first {j} tokens text: {repr(first_j_tokens)}')

    instruction = 'Continue the text in an autoregressive manner, starting from these words: '
    # print(tokenizer.encode(instruction, skip_special_tokens=True))
    prompt = instruction + start_token + first_j_tokens
    # print(f'prompt: {repr(prompt)}\n')
    messages = [
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
    outputs = model.generate(**input_ids, do_sample=True, max_new_tokens=2048)
    # outputs = model.generate(outputs1, do_sample=True, max_length=2048)
    gen_text = tokenizer.batch_decode(outputs[:, input_ids['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    gen_text = gen_text + start_token + first_j_tokens
    # gen_text = gen_text[len(prompt)+5:]
    # print(f'GENERATED: {repr(gen_text)}')
    # print('\nSET:', set(gen_text))
    uniq_length = len(set(gen_text))
    total_length = len(gen_text)
    print(f'Unique length: {uniq_length} Total length: {total_length}')
    if uniq_length < th or total_length < 256:
        ids_counter += 1
        print('Skipping...')
        continue
    # ids_counter += step_num
    # pbar.progress.update(pbar.task, advance=1)
    pbar.update(1)
    num_generated += 1

    text_dict = {"text" : gen_text}
    filename = f"gen_data/{MODEL_NAME}_gen.chunk.01.jsonl"
    with open(filename, "a") as f:
        print('writing to: ', filename)
        f.write(json.dumps(text_dict))
        f.write('\n')
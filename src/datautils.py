import os
import random
from typing import Optional
from pathlib import Path
import datasets
import numpy as np
import torch
from datasets import load_dataset
from packaging import version
from tqdm import trange
from transformers import AutoTokenizer, LlamaTokenizer
import json

def set_seed(seed: Optional[int]):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=False):
    print("Loading red_pajama from togethercomputer/RedPajama-Data-1T-Sample")
    assert not eval_mode, "Only train set is supported in RedPajama"
    traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train", trust_remote_code=True)
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    trainloader = []
    for _ in trange(nsamples, desc="Making red_pajama calibration set", leave=False):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
    return trainloader

def get_gsm8k(nsamples, seqlen, tokenizer, eval_mode=False):
    dataset = load_dataset('openai/gsm8k', split="train", trust_remote_code=True)
    dataset = dataset.filter(lambda example: len(example["response"]) >= seqlen)

    trainloader = []
    l = set()
    for data_item in dataset:
        template = "Question:\n{instruction}\n\nAnswer:\n{response}"
        # template = "{instruction}. {response}"
        train_data = template.format(**data_item)
        assert len(train_data) > seqlen, f'len={len(train_data)}'
        trainenc = tokenizer(train_data, return_tensors="pt").input_ids
        l.add(trainenc.numel())
        trainloader.append(trainenc[:, :seqlen])
    print(f'Loaded data: num_data={len(trainloader)}\n lengths={l}')
    return trainloader



def get_dolly(nsamples, seqlen, tokenizer, eval_mode=False):
    dataset = load_dataset('databricks/databricks-dolly-15k', split="train", trust_remote_code=True)
    dataset = dataset.filter(lambda example: len(example["response"]) >= seqlen)

    trainloader = []
    l = set()
    for i, data_item in enumerate(dataset):
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        # template = "{instruction}. {response}"
        train_data = template.format(**data_item)
        train_data += '\n\nInstruction:\n{instruction}\n\nResponse:\n'
        train_data = train_data.format(**dataset[i+1])
        assert len(train_data) > seqlen, f'len={len(train_data)}'
        trainenc = tokenizer(train_data, return_tensors="pt").input_ids
        # TODO: try chat template
        # prompt = instruction + start_token
        # messages = [
        #     {"role": "user", "content": train_data},
        # ]
        # trainenc = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        l.add(trainenc.numel())
        trainloader.append(trainenc[:, :seqlen])
    print(f'Loaded data: num_data={len(trainloader)}\n lengths={l}')
    return trainloader

def get_synthetic_as_is(model_id, tokenizer):
    ROOT_DIR = Path('/local_ssd2/nlyalyus/projects/aqlm/gen_data')
    model_name = Path(model_id).name.replace('.', '_')
    filename = f"{model_name}_gen.chunk.01.jsonl"
    dataset_path = ROOT_DIR / filename
    print("Loading syntehtic from ", dataset_path)
    trainloader = []
    for line in dataset_path.open("r"):
        text = json.loads(line)["text"]
        train_enc = tokenizer(text, return_tensors="pt").input_ids
        # TODO: need to pad, since concat is happening for batching, can run with batch=1
        # print('shape=', train_enc.shape)
        trainloader.append(train_enc)
    return trainloader


def get_synthetic_packed(nsamples, seqlen, tokenizer, model_id):
    ROOT_DIR = Path('/local_ssd2/nlyalyus/projects/aqlm/gen_data')
    model_name = Path(model_id).name.replace('.', '_')
    filename = f"{model_name}_gen.chunk.01.jsonl"
    dataset_path = ROOT_DIR / filename
    print("Loading syntehtic from ", dataset_path)
    concated_text = "\n\n".join(json.loads(line)["text"] for line in dataset_path.open("r"))
    train_enc = tokenizer(concated_text, return_tensors="pt").input_ids
    # print('shape=', train_enc.shape, ' nsamples=', train_enc.shape[1] // seqlen)
    trainloader = torch.split(train_enc, seqlen, dim=1)[:-1]
    return trainloader


def get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        print(type(trainenc), trainenc)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc


def get_ptb(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
    return testenc


def get_c4(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader

    else:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        random.seed(0)
        valenc = []
        for _ in range(256):
            while True:
                i = random.randint(0, len(valdata) - 1)
                tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
                if tmp.input_ids.shape[1] >= seqlen:
                    break
            if tmp.input_ids.shape[1] == seqlen:
                # rare case, discovered with Yi tokenizer
                valenc.append(tmp.input_ids)
            else:
                i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                valenc.append(tmp.input_ids[:, i:j])
        valenc = torch.hstack(valenc)
        return valenc


def get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        return testenc


def get_c4_new(nsamples, seqlen, tokenizer, eval_mode=False):
    if not eval_mode:
        traindata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            trainloader.append(inp)
        return trainloader
    else:
        valdata = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]
        return valenc


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    eval_mode=False,
    model_path=None,
    use_fast_tokenizer=False,
    trust_remote_code=None,
    model_id=None
):
    """
    Loads and prepares data for a Transformers model.
    Args:
        name (str): The name of the dataset to load.
        This can be one of 'wikitext2', 'c4', 'ptb','pajama' for datasets loaded from Huggingface datasets,
        or 'none' for cases where a dataset is not needed, like RTN. It can also accept data path to custom file.
        nsamples (int, optional): The number of samples to load from the dataset. Defaults to 128.
        seed (int, optional): The random seed value for data shuffling and splitting. Defaults to 0.
        seqlen (int, optional): The maximum sequence length for input tokenization. Defaults to 2048.
        model_path (str, optional): The path to the pretrained model weights or full model name.
            used to detect llama to call proper tokenizer.
            see https://github.com/huggingface/transformers/issues/22222#issuecomment-1488578722 for reasons.
        eval_mode (bool, optional). defines slice selection for 'wikitext2', 'c4', 'ptb' datasets.
        leave False for train slice.
        use_fast_tokenizer: whether to use fast tokenizer
        trust_remote_code: whether to trust remote code
    Returns:
        data (torch.utils.data.DataLoader or iterable): Data iterable for the dataset.
    Note:
        the popular decapoda-research Llama models have errors in tokenizer config, specifically
        incorrect token ids for BOS, EOS. This gets corrected to ensure compatibility with transformers
        of versions 4.29 and above.
    """
    set_seed(seed)

    # for pre-tokenized datasets

    if name.lower() == "none":
        print("Not loading any dataset. (OK if you use no compression or methods like RTN.)")
        return None
    elif os.path.isfile(name):
        try:
            data = torch.load(name)[:nsamples]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to load custom data from {name}.",
                "Check data path or use one of [c4, wikitext2, ptb, pajama, none]",
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=use_fast_tokenizer, trust_remote_code=trust_remote_code
        )

        if name.lower() == "wikitext2":
            data = get_wikitext2(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "pajama":
            data = get_red_pajama(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb":
            data = get_ptb(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "ptb_new":
            data = get_ptb_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4":
            data = get_c4(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "c4_new":
            data = get_c4_new(nsamples, seqlen, tokenizer, eval_mode=eval_mode)
        elif name.lower() == "synth":
            data = get_synthetic_packed(nsamples, seqlen, tokenizer, model_id)
        elif name.lower() == "dolly":
            data = get_dolly(nsamples, seqlen, tokenizer, model_id)
        else:
            raise ValueError(
                f"Failed to load data from {name}.",
                "Check dataset name or path or use one of [c4, wikitext2, ptb, pajama, none]",
            )

    if hasattr(data, "input_ids"):
        data = data.input_ids

    print(f"Loaded data from {name}; {len(data)=} sequences")
    return data

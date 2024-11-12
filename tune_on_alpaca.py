from typing import Dict, Optional, Tuple, List
import argparse
import os
import shutil
from copy import deepcopy
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn as nn
from tqdm import tqdm
import transformers
import numpy as np
import wandb

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)
from pathlib import Path

import random
import torch.nn as nn
from prompter import Prompter, ZeroPrompter

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_nncf_quantized_model(nncf_ckpt_dir, student_model, tokenizer):
    tokenized_text = tokenizer("example", return_tensors="pt")
    input_ids = tokenized_text["input_ids"]#[:, :-1]
    attention_mask = tokenized_text["attention_mask"]#[:, :-1]
    position_ids = torch.cumsum(attention_mask, axis=1) - 1
    position_ids[attention_mask == 0] = 1
    example_input = {
        "input_ids": input_ids.cuda(),
        "attention_mask": attention_mask.cuda(),
        "position_ids": position_ids.cuda()
    }
    print(example_input)
    nncf_ckpt = torch.load(Path(nncf_ckpt_dir) / 'nncf_checkpoint.pth')
    from nncf.torch import load_from_config

    # NOTE: hf_model.model because hf_model.model used for compression, where hf_model = AutoModelForCausalLM(...)
    student_model.model = load_from_config(
        student_model.model, nncf_ckpt["nncf_config"],
        example_input=example_input
    )
    student_model.model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])
    return student_model

def get_nb_trainable_parameters(module):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in module.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(module):
    trainable_params, all_param = get_nb_trainable_parameters(module)

    print(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


def generate_overfit(pipeline, tokenizer, prefix=""):
    messages = [
        {"role": "system", "content": "You can answer only with overfit word."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = pipeline.generate(
        inputs, min_new_tokens=32, max_new_tokens=32, do_sample=False
    )
    print("#" * 50 + f" {prefix}\n", tokenizer.decode(outputs[0]), "\n" + "#" * 150)

def kl_div(student_hiddens, teacher_hiddens):
    C = student_hiddens.shape[-1]  # num classes
    return F.kl_div(
        input=F.log_softmax(student_hiddens.view(-1, C), dim=-1),
        target=F.log_softmax(teacher_hiddens.view(-1, C), dim=-1),
        log_target=True,
        reduction="batchmean",
    )

def compute_validation_perplexities(args: argparse.Namespace, model: nn.Module, eval_datasets):
    perplexities = {}
    for dataset_name, eval_dataset in eval_datasets.items():
        print(f"Evaluating perplexity on {dataset_name} ...")
        device = next(model.parameters()).device
        original_dtype = args.load_dtype if args.load_dtype != "auto" else None
        amp_dtype = args.amp_dtype if args.amp_dtype is not None else original_dtype
        ppl = evaluate_perplexity(model, eval_dataset, args.model_seqlen, device=device, amp_dtype=amp_dtype)
        print(f"{dataset_name} perplexity: {ppl:.9f}")
        perplexities[dataset_name] = ppl
    return perplexities


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module, data: torch.Tensor, seqlen: int, device: torch.device, amp_dtype: Optional[torch.dtype] = None
) -> float:
    """Perplexity evaluation as per https://github.com/IST-DASLab/gptq (standard among quantization research)"""
    inps = [
        data[:, start : start + seqlen] for start in range(0, data.shape[1], seqlen) if start + seqlen < data.shape[1]
    ]  # ignore last incomplete sequence as in the GPTQ paper
    num_sequences_without_padding = len(inps)

    # NOTE: world_size = 1 -> num_padding_sequences=0
    # # pad sequences to be divisible by world_size for DDP/FSDP compatibility
    # num_padding_sequences = -len(inps) % world_size
    # inps.extend([inps[-1]] * num_padding_sequences)

    total_nll_and_tokens = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
    total_nll, total_tokens = total_nll_and_tokens[0], total_nll_and_tokens[1]

    for sequence_index, input_ids in enumerate(tqdm(inps, desc="Evaluating perplexity")):
        input_ids = input_ids.to(device)
        with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
            lm_logits = model(input_ids).logits

        if sequence_index < num_sequences_without_padding:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_nll += loss.float() * shift_labels.numel()
            total_tokens += shift_labels.numel()

    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()

def set_trainable(args, model, list_of_indexes: List[int]):
    for param in model.parameters():
        param.requires_grad = False

    diff_params = {}
    adapters_to_train = []
    scales_to_train = []

    # word_to_find = "lora" if is_lora else "input"
    for name, param in model.named_parameters():
        for index in list_of_indexes:
            # if f"layers_{index}_" in name and word_to_find in name:
            # if f"layers_{index}_" in name and "lora" in name:
            #     param.requires_grad = True
            #     adapters_to_train.append(param)
            #     break
            if f"layers_{index}_" in name and "input" in name:
                param.requires_grad = True
                scales_to_train.append(param)
                break

    for name, param in model.named_parameters():
        if param.requires_grad:
            print('Tune: ', name)
    # num_grad = sum(map(lambda x: x.requires_grad, model.parameters()))
    # num_lora = sum(map(lambda x: "lora" in x[0], model.named_parameters()))
    # assert num_lora == num_grad, f"number of lora params != number of learnable params ({num_lora} vs {num_grad})"
    print_trainable_parameters(model)

    param_to_train = [
        # {"params": adapters_to_train,  "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": scales_to_train, "lr": args.fq_lr, "weight_decay": args.weight_decay}
    ]
    # diff_params = TODO: merge dicts
    return param_to_train

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def print_memory_stats():
    print(f"GPU max memory allocated: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB.")
    print(f"GPU max memory reserved: {torch.cuda.max_memory_reserved() / 2 ** 30:.2f} GB.")

def save_checkpoint(wrapped_model, ckpt_dir):
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    nncf_state_dict = wrapped_model.nncf.state_dict()
    nncf_config = wrapped_model.nncf.get_config()
    torch.save(
        {
            "nncf_state_dict": nncf_state_dict,
            "nncf_config": nncf_config,
        },
        ckpt_dir / "nncf_checkpoint.pth",
    )

def get_model(
    model_path, dtype="auto", device_map=None, attn_implementation=None, trust_remote_code=False
):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code).torch_dtype or "auto"
        )  # force transformers 4.29.2 to follow the same rules as 4.30.x
    else:
        dtype = getattr(torch, dtype)

    model_kwargs = {}
    # this argument is avaialbe only for transformers >= 4.38.0
    if transformers.__version__ >= "4.38.0":
        model_kwargs["attn_implementation"] = attn_implementation

    print('nlyalyus: device_map', device_map)
    orig_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        # defer distribution if loading quantized
        device_map=device_map,
        low_cpu_mem_usage=True,
        local_files_only=True,
        **model_kwargs,
    )

    # if not args.device_map:
    #     orig_model = orig_model.to(device)
    # compute_validation_perplexities(args, orig_model, eval_datasets)
    # generate_overfit(orig_model, tokenizer, "FP32")

    quant_model = load_nncf_quantized_model(args.nncf_ckpt_dir, orig_model, tokenizer)
    # generate_overfit(quant_model, tokenizer, "FQ")
    # compute_validation_perplexities(args, quant_model, eval_datasets)
    # if not args.device_map:
    #     device = "cuda"
    #     quant_model = quant_model.to(device)

    print("Model loaded sucсessfully ...")
    return quant_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    # Model params
    parser.add_argument(
        "--fq_lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--lr_scale",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default=None,
        help="if specified, runs automated mixed precision with this dtype",
    )
    parser.add_argument(
        "--load_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--print_every_steps",
        type=int,
        default=None,
        help="print training metrics once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=None,
        help="evaluate once in this many optimizer steps (this many updates to model parameters)",
    )
    parser.add_argument(
        "--nncf_ckpt_dir",
        type=str,
        required=False,
        help="path to quantized model",
    )
    # Data params
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024,
        help="number of samples",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--eval_model_seqlen",
        type=int,
        default=None,
        help="Model seqlen on validation. By default is equal to model_seqlen.",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="size of validation split",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2"],#, "c4"],
        help="Datasets to run evaluation on",
    )
    parser.add_argument("--keep_best_model", action="store_true", help="Save best model state separately")
    parser.add_argument("--skip_first_eval", action="store_true", default=None)

    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)

    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")

    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="finetuning learning rate",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.90,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="Adam beta2",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="training microbatch size",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to apply gradient checkpointing",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Whether to use amp",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=3,
        help="Terminate finetuning if loss doesn't improve after this number of epochs.",
    )
    parser.add_argument(
        "--finetune_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to finetune the model",
    )
    # Logging params
    parser.add_argument('--wandb_project', type=str, default="")
    # Save params
    parser.add_argument("--exp_name", type=str, default=None, help="Path to save quantized models.")
    # Misc params
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload_activations", action="store_true", help="Offload activations to RAM to save GPU memory.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=[None, "auto"],
        help="accelerate device map",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    #ddp
    # parser.add_argument('--local_rank', type=int, default=-1)

    set_seed(42)
    args = parser.parse_args()

    args.microbatch_size = args.microbatch_size or args.batch_size
    args.finetune_dtype = getattr(torch, args.finetune_dtype)
    if args.amp:
        assert args.finetune_dtype == torch.float32, "AMP works only with original model in fp32."

    # get device
    assert torch.cuda.is_available()

    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=args.use_fast_tokenizer, trust_remote_code=True
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if 'lamini' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["response"],
            )
        elif 'alpaca' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
        else:
            raise NotImplementedError

        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set

    # TODO:
    # if device == 'cuda':
    #     model.half()

    quant_model = get_model(args.base_model, args.dtype, args.device_map, trust_remote_code=args.trust_remote_code)

    ckpt_dir = Path(args.nncf_ckpt_dir) / args.exp_name
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Load Train Dataset
    data = load_dataset(args.data_path)
    if args.cache_dataset and os.path.exists('datasets/cache/{}.bin'.format(args.data_path)):
        preprocess_data = torch.load('datasets/cache/{}.bin'.format(args.data_path))
        train_data, val_data = preprocess_data['train'], preprocess_data['val']
    else:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = {
            args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
        }
        if args.cache_dataset:# and args.local_rank == 0:
            cache_file = 'datasets/cache/{}.bin'.format(args.data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)

            torch.save({
                'train': train_data, 'val': val_data
            }, cache_file)

    # trainer = transformers.Trainer(
    #     model=quant_model,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=args.micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=100,
    #         num_train_epochs=args.num_epochs,
    #         learning_rate=args.learning_rate,
    #         fp16=True,
    #         logging_steps=10,
    #         logging_first_step=True,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps",
    #         save_strategy="steps",
    #         eval_steps=100,
    #         save_steps=200,
    #         output_dir=args.output_dir,
    #         save_total_limit=20,
    #         load_best_model_at_end=True,
    #         ddp_find_unused_parameters=None,
    #         group_by_length=args.group_by_length,
    #         report_to="wandb",
    #         run_name=args.output_dir.split('/')[-1],
    #         metric_for_best_model="{}_loss".format(args.data_path),
    #     ),
    #     data_collator=transformers.DataCollatorForSeq2Seq(
    #         tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     ),
    # )
    # model.config.use_cache = False

    # trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # # compute_validation_perplexities(args, quant_model, eval_datasets)
    # last_dir = ckpt_dir / "last_ckpt"
    # last_dir.mkdir(exist_ok=True, parents=True)
    # save_checkpoint(quant_model.model, last_dir)

    # print_memory_stats()

    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    # print(f"eval: {torch.cuda.max_memory_allocated()=:,}")

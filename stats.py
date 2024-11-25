from typing import Dict, Optional, Tuple, List
import argparse
import os
import shutil
from copy import deepcopy
import time
import torch
import torch.nn.functional as F
from torch import nn as nn
from accelerate.hooks import remove_hook_from_submodules
from tqdm import tqdm, trange
from nncf.torch.dynamic_graph.context import forward_nncf_trace
import transformers
import numpy as np
from collections import defaultdict
try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from main import perplexity_eval
from src.datautils import get_loaders
from src.modelutils import get_layers, get_model, save_not_quantized_weights
from src.utils import _extract_into_tensor, maybe_get_0th_element
from transformers import (
    AutoTokenizer
)
from pathlib import Path
from whowhatbench import Evaluator
import random
import torch.nn as nn

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

# Function to register hooks and collect activations
def register_hooks(model):
    mean_values_list_per_module = defaultdict(list)

    def hook_fn(module, input_: torch.Tensor, output: torch.Tensor):
        """
        X - average channel magnitude across tokens in the sequence [HiddenDim, min(SampleSize, ~subset_size)]
        X = fns.stack(stats.mean_values)  # [SampleSize, HiddenDim]
        X_full = fns.transpose(X)  # [HiddenDim, SampleSize]
        """
        if isinstance(module, nn.Linear):
            # TODO: take the right axis [1, Seq Length, Hidden Dim]
            reduction_axes = 1
            mean_across_tokens = torch.mean(input_[0], dim=(reduction_axes,), keepdim=True)
            mean_values_list_per_module[module].append(mean_across_tokens)

    for layer in model.modules():
        layer.register_forward_hook(hook_fn)

    return mean_values_list_per_module

def get_Xmap(model_):
    mean_values_list_per_module = register_hooks(model_)
    for i in trange(len(calib_dataloader), total=len(calib_dataloader), desc="Caching activations", leave=False):
        batch = maybe_get_0th_element(calib_dataloader[i]).to(device)
        model_.model(batch)

    def stack_values(item):
        key, value = item
        return (key, torch.stack(value).squeeze())

    mean_values_list_per_module = dict(map(stack_values, mean_values_list_per_module.items()))
    # for layer, mean_act in mean_values_list_per_module.items():
    #     # X - average channel magnitude across tokens in the sequence [HiddenDim, min(SampleSize, ~subset_size)]
    #     # X = fns.stack(stats.mean_values)  # [SampleSize, HiddenDim]
    #     # X_full = fns.transpose(X)  # [HiddenDim, SampleSize]
    #     XW = torch.stack(mean_act).squeeze() @ layer.weight.t()
    #     print(f"Layer: {layer}, Mean Activation: shape={XW.shape}")
    return mean_values_list_per_module

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
        type=int,
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
    # parser.add_argument(
    #     "--quant_model",
    #     type=str,
    #     required=True,
    #     help="path to quantized model",
    # )
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
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    # Save params
    parser.add_argument("--exp_name", type=str, default=None, help="Path to save quantized models.")
    # Misc params
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
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
    args = parser.parse_args()

    device = "cuda"
    args.devices = [device]  # needed for perplexity eval

    calib_dataloader = get_loaders(
        args.dataset,
        nsamples=5,
        seed=32,
        model_path=args.base_model,
        seqlen=args.model_seqlen,
        use_fast_tokenizer=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    # create original model
    orig_model = get_model(args.base_model, None, args.dtype, args.device_map, trust_remote_code=args.trust_remote_code)
    if not args.device_map:
        orig_model = orig_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=args.use_fast_tokenizer, trust_remote_code=True
    )

    Xmap = get_Xmap(orig_model)

    quant_model = load_nncf_quantized_model(args.nncf_ckpt_dir, orig_model, tokenizer)
    print(quant_model)

    def get_quant_noise(quant_model, Xmap):
        loss = 0
        for name, quantizer in quant_model.model._nncf.external_quantizers.items():
            # layers_31_mlp_gate_up_proj_weight
            base_list = name.split('FQ_LORA_for_node_layers_')[1].split('_')
            layer_id = int(base_list[0])
            next_id = 2
            if base_list[1] == 'mlp':
                sub_block = base_list[1]
            else:
                sub_block = '_'.join(base_list[1:3])
                next_id = 3
            layer_name = '_'.join(base_list[next_id:-1])
            sub_block = getattr(quant_model.model.layers[layer_id], sub_block)
            layer = getattr(sub_block, layer_name)
            X = Xmap[layer]
            W = layer.weight.data
            FQ_W = quantizer.quantize(W)
            diff = X @ W.t() - X @ FQ_W.t()
            loss_ = torch.linalg.norm(diff, ord="fro").item()
            print('layer={} loss={:.1f}'.format(name, loss_))
            loss += loss_
        return loss

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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_quant_noise(quant_model, Xmap):
    loss = 0
    for name, quantizer in quant_model.model._nncf.external_quantizers.items():
        # layers_31_mlp_gate_up_proj_weight
        # layers_22_self_attn_q_proj_weight
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
        W = layer.weight
        FQ_W = quantizer.quantize(W)
        diff = X @ W.t() - X @ FQ_W.t()
        loss_ = torch.linalg.norm(diff, ord="fro").item()
        # print('layer={} loss={:.1f}'.format(name, loss_))
        loss += loss_
    return loss

def get_Xmap(model_, calib_dataloader):
    mean_values_list_per_module = register_hooks(model_)
    for i in trange(len(calib_dataloader), total=len(calib_dataloader), desc="Collecting activations", leave=False):
        batch = maybe_get_0th_element(calib_dataloader[i]).to(device)
        model_.model(batch)

    def stack_values(item):
        key, value = item
        return (key, torch.stack(value).squeeze())

    mean_values_list_per_module = dict(map(stack_values, mean_values_list_per_module.items()))
    return mean_values_list_per_module

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
            # TODO: take the right axis. 0? [Seq Length, Hidden Dim]
            # print(f'W={module.weight.shape} X={input_[0].shape} O={output.shape}')
            reduction_axes = 1
            mean_across_tokens = torch.mean(input_[0], dim=(reduction_axes,), keepdim=True)
            mean_values_list_per_module[module].append(mean_across_tokens)

    for layer in model.modules():
        layer.register_forward_hook(hook_fn)

    return mean_values_list_per_module



@torch.inference_mode()
def cache_hiddens(model, dataloader):
    device = next(model.parameters()).device
    cached_hiddens = []
    for i in trange(len(dataloader), total=len(dataloader), desc="Caching hiddens", leave=False):
        with torch.autocast(device_type="cuda", enabled=args.amp):
            batch = maybe_get_0th_element(dataloader[i]).to(device)
            cached_hiddens.append(model.model(batch).last_hidden_state.cpu())
    return cached_hiddens


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
        # TODO: hack amp_dtype or torch.float32):
        with torch.amp.autocast('cuda', enabled=amp_dtype is not None, dtype=torch.bfloat16):
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
    B_adapters_to_train = []
    scales_to_train = []

    # word_to_find = "lora" if is_lora else "input"
    for name, param in model.named_parameters():
        for index in list_of_indexes:
            # if f"layers_{index}_" in name and word_to_find in name:
            if f"layers_{index}_" in name and "lora_A" in name:
                param.requires_grad = True
                adapters_to_train.append(param)
                break
            if f"layers_{index}_" in name and "lora_B" in name:
                param.requires_grad = True
                B_adapters_to_train.append(param)
                break
            if f"layers_{index}_" in name and "input" in name:
                param.requires_grad = True
                scales_to_train.append(param)
                break

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print('Tune: ', name)
    print_trainable_parameters(model)

    param_to_train = [
        {"params": adapters_to_train,  "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": B_adapters_to_train,  "lr": 10 * args.lr, "weight_decay": args.weight_decay},
        {"params": scales_to_train, "lr": args.fq_lr, "weight_decay": 0}#args.weight_decay}
    ]
    return param_to_train

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def finetune(model, train_loader, train_hiddens, args, device, ckpt_dir=None, eval_datasets=None, Xmap=None):
    # cast model to finetune dtype
    model.to(args.finetune_dtype)
    # NOTE: copy is needed for calculating target outputs
    lm_head = deepcopy(model.lm_head)
    for param in lm_head.parameters():
        param.requires_grad = False

    grad_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    metadata = dict(
        current_epoch=0,
        microbatches_since_epoch_start=0,
        total_microbatches=0,
        total_optimizer_steps=0,
        kl_loss_numerator=0,
        q_loss_numerator=0,
        loss_denominator=0,
        aggregated_loss=float("nan"),
        aggregated_kl_loss=float("nan"),
        aggregated_q_loss=float("nan"),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float("inf"),
        best_step=0,
    )
    NUM_LAYERS = 32
    HIDDEN_DIM = 3072
    num_switches = NUM_LAYERS // args.num_blocks
    num_epochs = num_switches * args.frequency
    # num_epochs = args.epochs
    num_training_steps = epoch_samples // args.batch_size
    cached_targets = None
    layer = quant_model.model._nncf.external_quantizers.FQ_LORA_for_node_layers_23_mlp_down_proj_weight
    first_loss = 0.07
    first_qloss = 1322
    ratio = 1/2
    factor = first_qloss / first_loss
    for epoch in range(num_epochs):
        if epoch % args.frequency == 0:
            i_switch = epoch // args.frequency
            metadata["num_tuned_blocks"] = i_switch * args.num_blocks
            active_layers_ids = list(range(i_switch * args.num_blocks, (i_switch + 1) * args.num_blocks))
            # active_layers_ids = list(range(23,24))
            param_to_train = set_trainable(args, model, active_layers_ids)
            if not param_to_train:
                print('All layers are tuned!')
                break

            # Slightly un-intuitive but we want to increase the rate as the layers progress
            # because error accumulates and we want to correct it more strongly.
            # scaled_lr = args.lr * (1 + args.lr_scale * (i_switch / num_switches))
            # scaled_lr = scaled_lr if is_lora else scaled_lr / 50
            # metadata["scaled_lr"] = scaled_lr
            opt = torch.optim.AdamW(param_to_train, betas=(args.adam_beta1, args.adam_beta2))
            # scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
            if args.cosine:
                lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                    opt,
                    num_warmup_steps=args.warmup,
                    num_training_steps=num_epochs * num_training_steps,
                    num_cycles=0.5
                )
            else:
                lr_scheduler = transformers.get_constant_schedule_with_warmup(opt, num_warmup_steps=args.warmup)

        # train loop
        model.train()
        # prepare batch indices
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)

        for batch_indices in tqdm(batch_indices_epoch, desc=f"Train epoch {epoch}", leave=False):
            # convert tensor to list
            batch_indices = batch_indices.tolist()
            metadata["microbatches_since_epoch_start"] += 1
            metadata["total_microbatches"] += 1

            inputs = _extract_into_tensor(train_loader, batch_indices, device=device)
            with torch.no_grad():
                targets = lm_head(
                    _extract_into_tensor(train_hiddens, batch_indices, device=device, dtype=args.finetune_dtype)
                )

            # with torch.autocast(device_type="cuda", enabled=args.amp):
            outputs = model(inputs).logits
            kl_loss = kl_div(outputs, targets.to(device=outputs.device, dtype=args.finetune_dtype))
            q_loss = 0 if Xmap is None else get_quant_noise(model, Xmap)
            loss = kl_loss + ratio * q_loss / factor
            metadata["kl_loss_numerator"] += kl_loss.item()
            metadata["q_loss_numerator"] += q_loss
            metadata["loss_denominator"] += 1
            metadata["grad_steps_accumulated"] += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")

            # A_before = quant_model.model._nncf.external_quantizers.FQ_LORA_for_node_layers_23_mlp_down_proj_weight._lora_A.data.clone()
            # scaler.scale(loss / grad_accumulation_steps).backward()
            (loss / grad_accumulation_steps).backward()

            metadata["23dj_gA"] = torch.linalg.norm(layer._lora_A.grad.data).item()
            metadata["23dj_gB"] = torch.linalg.norm(layer._lora_B.grad.data).item()
            metadata["23dj_gIL"] = torch.linalg.norm(layer.input_low.grad.data).item()
            metadata["23dj_gIR"] = torch.linalg.norm(layer.input_range.grad.data).item()

            if metadata["grad_steps_accumulated"] == grad_accumulation_steps:
                # print('step')
                lr_scheduler.step()
                metadata["lr"] = get_lr(opt)
                opt.step()
                # scaler.step(opt)
                # scaler.update()
                opt.zero_grad()
                # reset accumulated step and loss
                metadata["grad_steps_accumulated"] = 0
                metadata["total_optimizer_steps"] += 1
                metadata["aggregated_kl_loss"] = metadata["kl_loss_numerator"] / metadata["loss_denominator"]
                metadata["aggregated_q_loss"] = ratio * metadata["q_loss_numerator"] / metadata["loss_denominator"] / factor
                # if epoch == 0 and metadata["total_optimizer_steps"] == 1:
                #     first_loss = metadata["aggregated_kl_loss"]
                #     first_qloss = factor * metadata["aggregated_q_loss"] / ratio
                #     ratio = 1/2
                #     factor = first_qloss/first_loss
                metadata["aggregated_loss"] = metadata["aggregated_kl_loss"] + metadata["aggregated_q_loss"]
                metadata["kl_loss_numerator"] = metadata["loss_denominator"] = metadata["q_loss_numerator"] = 0

                metadata["23dj_A"] = torch.linalg.norm(layer._lora_A.data).item()
                metadata["23dj_B"] = torch.linalg.norm(layer._lora_B.data).item()
                metadata["23dj_IL"] = torch.linalg.norm(layer.input_low.data).item()
                metadata["23dj_IR"] = torch.linalg.norm(layer.input_range.data).item()
                # visualize norm A, B, input_low, input_range,
                # layer=FQ_LORA_for_node_layers_23_mlp_down_proj_weight - 142
                # layer=FQ_LORA_for_node_layers_23_self_attn_o_proj_weight - 35

            # NOTE: debug that some parameters updated and some - frozen
            # A_after = quant_model.model._nncf.external_quantizers.FQ_LORA_for_node_layers_31_mlp_gate_up_proj_weight._lora_A.data
            # assert torch.equal(W_after, W_after)

            if args.print_every_steps and metadata["total_optimizer_steps"] % args.print_every_steps == 0 and metadata["grad_steps_accumulated"] == 0:
                print(
                    f"epoch {metadata['current_epoch']}\t", # TODO: batch index and re-do trainloder??
                    f"\t| total updates = {metadata['total_optimizer_steps']}",
                    f"\tkl_loss = {metadata['aggregated_loss']:.9f}",
                    f"\tq_loss = {metadata['aggregated_q_loss']:.9f}",
                    f"\tloss = {metadata['aggregated_kl_loss']:.9f}",
                )

            if args.wandb:
                wandb.log(metadata, step=metadata["total_microbatches"])

        # TODO: why removed eval loss ??? maybe needed??? quick and on test data
        # TODO: run lm_eval on wikitext??
        # NOTE: evaluate in the end of each epoch
        # if (epoch + 1) % args.frequency == 0:
        # TODO: eval similarity!? with chat template
        perplexity_scores = compute_validation_perplexities(args, model, eval_datasets)
        for dataset_name, perplexity in perplexity_scores.items():
            metadata[f"perplexity_{dataset_name}"] = perplexity
        metric_name = metadata["early_stop_on"]
        if perplexity_scores[metric_name] < metadata["best_eval_perplexity"]:
            print(f"New best perplexity ({metric_name}) = {perplexity_scores[metric_name]:.9f}")
            metadata["best_eval_perplexity"] = perplexity_scores[args.eval_datasets[0]]
            metadata["best_step"] = metadata["total_optimizer_steps"]
            if args.keep_best_model:
                save_checkpoint(model.model, ckpt_dir)

        metadata["microbatches_since_epoch_start"] = 0
        metadata["current_epoch"] += 1


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
        "--cosine",
        action="store_true",
        default=None,
        help="Whether to use fast tokenizer.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    set_seed(42)
    args = parser.parse_args()
    if args.keep_best_model:
        assert args.eval_datasets is not None, f"--keep_best_model requires --eval_datasets"


    model_name = Path(args.base_model).name.replace('.', '_')
    ROOT_MODEL_DIR = Path.home() / ("MODEL_DIR")
    MODEL_DIR = ROOT_MODEL_DIR / model_name
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    args.microbatch_size = args.microbatch_size or args.batch_size
    args.finetune_dtype = getattr(torch, args.finetune_dtype)
    if args.amp:
        assert args.finetune_dtype == torch.float32, "AMP works only with original model in fp32."

    # get device
    assert torch.cuda.is_available()
    device = "cuda"
    args.devices = [device]  # needed for perplexity eval
    is_cosine = 'cosine' if args.cosine else 'const'
    exp_name =  args.exp_name if args.exp_name else f"slm_{is_cosine}_both_g64_rank256_lr{args.lr:.0e}_n{args.nsamples}_fqlr{args.fq_lr:.0e}_wd{args.weight_decay:.0e}_bs{args.batch_size}_rand100+_qloss_10xB"
    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args, name=exp_name)

    # get data
    train_dataloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.base_model,
        seqlen=args.model_seqlen,
        use_fast_tokenizer=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    calib_dataloader = get_loaders(
        args.dataset,
        nsamples=12,
        seed=args.seed,
        model_path=args.base_model,
        seqlen=args.model_seqlen,
        use_fast_tokenizer=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    eval_datasets = {
        dataset_name: get_loaders(
            dataset_name,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=args.model_seqlen,
            eval_mode=True,
        )
        for dataset_name in args.eval_datasets
    }

    # create original model
    orig_model = get_model(args.base_model, None, args.dtype, args.device_map, trust_remote_code=args.trust_remote_code)
    if not args.device_map:
        orig_model = orig_model.to(device)
    # compute_validation_perplexities(args, orig_model, eval_datasets)
    # exit()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=args.use_fast_tokenizer, trust_remote_code=True
    )

    # NOTE: overfit experiments
    # train_dataloader = [tokenizer("b", return_tensors="pt")["input_ids"]]

    # cache logits
    start = time.time()
    CACHE_DIR = MODEL_DIR / 'hiddens_cache'
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    TRAIN_HIDDENS_PATH = CACHE_DIR / Path(f'{model_name}_train_hiddens_n{args.nsamples}_data_{args.dataset}.pth')
    # TRAIN_HIDDENS_PATH = CACHE_DIR / Path('blabla.pth')
    if TRAIN_HIDDENS_PATH.exists():
        orig_train_hiddens = torch.load(TRAIN_HIDDENS_PATH)
    else:
        orig_train_hiddens = cache_hiddens(orig_model, train_dataloader)
        torch.save(orig_train_hiddens, TRAIN_HIDDENS_PATH)
    # del orig_model
    # torch.cuda.empty_cache() # TODO:???
    Xmap = None
    # Xmap = get_Xmap(orig_model, calib_dataloader)
    print(f'Caching took {(time.time() - start):.2f} seconds')

    # generate_overfit(orig_model, tokenizer, "FP32")
    wwb_ref = MODEL_DIR / "ref_qa.csv"
    if not wwb_ref.exists():
        evaluator = Evaluator(base_model=orig_model, tokenizer=tokenizer, metrics=("similarity",))
        evaluator.dump_gt(str(wwb_ref))

    quant_model = load_nncf_quantized_model(args.nncf_ckpt_dir, orig_model, tokenizer)
    # generate_overfit(quant_model, tokenizer, "FQ")
    # compute_validation_perplexities(args, quant_model, eval_datasets)
    if not args.device_map:
        quant_model = quant_model.to(device)

    ckpt_dir = Path(args.nncf_ckpt_dir) / exp_name
    ckpt_dir.mkdir(exist_ok=True, parents=True)


    # finetune
    finetune(
        quant_model,
        train_loader=train_dataloader,
        train_hiddens=orig_train_hiddens,
        args=args,
        device=device,
        ckpt_dir=ckpt_dir,
        eval_datasets=eval_datasets,
        Xmap=Xmap
    )

    # generate_overfit(quant_model, tokenizer, "after tune")

    # compute_validation_perplexities(args, quant_model, eval_datasets)
    # last_dir = ckpt_dir / "last_ckpt"
    # last_dir.mkdir(exist_ok=True, parents=True)
    # save_checkpoint(quant_model.model, last_dir)

    print_memory_stats()

    # TODO: why offload model to cpu??
    # quant_model = quant_model.cpu()
    # TODO: why is this accelerate function needed? can delete FQ?
    # if args.device_map:
    #     remove_hook_from_submodules(quant_model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})

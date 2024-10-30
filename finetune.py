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

@torch.inference_mode()
def cache_hiddens(model, dataloader):
    device = next(model.parameters()).device
    cached_hiddens = []
    for i in trange(len(dataloader), total=len(dataloader), desc="Caching hiddens", leave=False):
        with torch.autocast(device_type="cuda", enabled=args.amp):
            batch = maybe_get_0th_element(dataloader[i]).to(device)
            cached_hiddens.append(model.model(batch).last_hidden_state.cpu())
    return cached_hiddens


def generate_chicken(pipeline, tokenizer, prefix=""):
    output = pipeline.generate(
        tokenizer("chicken", return_tensors="pt")["input_ids"].cuda(), min_new_tokens=32, max_new_tokens=32
    )
    print("#" * 50 + f" {prefix}\n", tokenizer.decode(output[0]), "\n" + "#" * 150)

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
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32):
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

def set_trainable(model, list_of_indexes: List[int]):
    for param in model.parameters():
        param.requires_grad = False

    diff_params = {}
    for name, param in model.named_parameters():
        for index in list_of_indexes:
            if f"layers_{index}_" in name and "lora" in name:
                print('Tune: ', name)
                param.requires_grad = True
                diff_params[name] = param
                break

    # num_grad = sum(map(lambda x: x.requires_grad, model.parameters()))
    # num_lora = sum(map(lambda x: "lora" in x[0], model.named_parameters()))
    # assert num_lora == num_grad, f"number of lora params != number of learnable params ({num_lora} vs {num_grad})"
    print_trainable_parameters(model)

    return diff_params


def finetune(model, train_loader, train_hiddens, args, device, val_loader=None, val_hiddens=None, ckpt_dir=None, eval_datasets=None):
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

    # compute_validation_perplexities(args, model, eval_datasets)

    metadata = dict(
        current_epoch=0,
        microbatches_since_epoch_start=0,
        total_microbatches=0,
        total_optimizer_steps=0,
        loss_numerator=0,
        loss_denominator=0,
        aggregated_loss=float("nan"),
        grad_steps_accumulated=0,
        early_stop_on=next(iter(args.eval_datasets)) if args.eval_datasets else None,
        best_eval_perplexity=float("inf"),
        best_step=0,
    )

    eval_step = -1
    NUM_LAYERS = 5
    FREQUENCY = 3
    for epoch in range(args.epochs):
        if epoch % FREQUENCY == 0:
            num_iters = epoch // FREQUENCY
            active_layers_ids = list(range(num_iters * NUM_LAYERS, (num_iters + 1) * NUM_LAYERS))
            # active_layers_ids = list(range(30,32))
            diff_params = set_trainable(model, active_layers_ids)
            if not diff_params:
                print('All layers are tuned!')
                break
            opt = torch.optim.Adam(diff_params.values(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
            scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

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

            with torch.autocast(device_type="cuda", enabled=args.amp):
                outputs = model(inputs).logits
            loss = kl_div(outputs, targets.to(device=outputs.device, dtype=args.finetune_dtype))

            metadata["loss_numerator"] += loss.item()
            metadata["loss_denominator"] += 1
            metadata["grad_steps_accumulated"] += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")

            scaler.scale(loss / grad_accumulation_steps).backward()

            if metadata["grad_steps_accumulated"] == grad_accumulation_steps:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                # reset accumulated step and loss
                metadata["grad_steps_accumulated"] = 0
                metadata["total_optimizer_steps"] += 1
                metadata["aggregated_loss"] = metadata["loss_numerator"] / metadata["loss_denominator"]
                metadata["loss_numerator"] = metadata["loss_denominator"] = 0

            if args.print_every_steps and metadata["total_optimizer_steps"] % args.print_every_steps == 0 and metadata["grad_steps_accumulated"] == 0:
                print(
                    f"epoch {metadata['current_epoch']}\t", # TODO: batch index and re-do trainloder??
                    f"\t| total updates = {metadata['total_optimizer_steps']}",
                    f"\tloss = {metadata['aggregated_loss']:.9f}",
                )

            # TODO: make no eval for 0-fiteration each time!
            if args.eval_every_steps and \
                metadata["total_optimizer_steps"] % args.eval_every_steps == 0 and \
                    metadata["total_optimizer_steps"] != eval_step and \
                        not(args.skip_first_eval and metadata["total_optimizer_steps"] == 0):
                eval_step = metadata["total_optimizer_steps"]
                # TODO: why removed eval loss ??? maybe needed??? quick and on test data
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

            if args.wandb:
                wandb.log(metadata, step=metadata["total_microbatches"])

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

# def evaluate_model(model_to_eval, args):
#     """
#     NOTE: NNCF doesn't work in layer-wise setup!
#     Can infer via NNCFNetwork with enabled tracing on forward.
#     """

#     for dataset in args.eval_datasets:
#         testloader = get_loaders(
#             dataset,
#             seed=args.seed,
#             model_path=args.base_model,
#             seqlen=args.eval_model_seqlen or args.model_seqlen,
#             eval_mode=True,
#             use_fast_tokenizer=args.use_fast_tokenizer,
#             trust_remote_code=args.trust_remote_code,
#         )
#         args.dataset_name = dataset
#         perplexity_eval(model_to_eval, testloader, args)
#     torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    # Model params
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
    if args.keep_best_model:
        # assert args.exp_name is not None, f"--keep_best_model requires --exp_name path"
        assert args.eval_every_steps is not None, f"--keep_best_model requires --eval_every_steps"
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
    exp_name = args.exp_name if args.exp_name else f"n{args.nsamples}"
    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args, name=exp_name)

    # get data
    dataloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.base_model,
        seqlen=args.model_seqlen,
        use_fast_tokenizer=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if args.val_size > 0:
        all_ids = torch.randperm(len(dataloader))
        train_ids, val_ids = all_ids[args.val_size :], all_ids[: args.val_size]
        train_dataloader = [dataloader[i] for i in train_ids]
        val_dataloader = [dataloader[i] for i in val_ids]
    else:
        train_dataloader = dataloader
        val_dataloader = None

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
    generate_chicken(orig_model, tokenizer, "FP32")
    # exit()
    # evaluate_model(orig_model, args)

    # cache logits
    start = time.time()
    CACHE_DIR = MODEL_DIR / 'hiddens_cache'
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    TRAIN_HIDDENS_PATH = CACHE_DIR / Path(f'{model_name}_train_hiddens_n{args.nsamples}_data_{args.dataset}.pth')
    VAL_HIDDENS_PATH = CACHE_DIR / Path(f'{model_name}_val_hiddens_n{args.nsamples}_data_{args.dataset}.pth')
    if TRAIN_HIDDENS_PATH.exists():
        orig_train_hiddens = torch.load(TRAIN_HIDDENS_PATH)
    else:
        orig_train_hiddens = cache_hiddens(orig_model, train_dataloader)
        torch.save(orig_train_hiddens, TRAIN_HIDDENS_PATH)
    if val_dataloader:
        if VAL_HIDDENS_PATH.exists():
            orig_val_hiddens = torch.load(VAL_HIDDENS_PATH)
        else:
            orig_val_hiddens = cache_hiddens(orig_model, val_dataloader)
            torch.save(orig_val_hiddens, VAL_HIDDENS_PATH)
    else:
        orig_val_hiddens = None
    # del orig_model
    # torch.cuda.empty_cache() # TODO:???
    print(f'Caching took {(time.time() - start):.2f} seconds')

    # load NNCF model with FQ's
    # TODO: probably nothing bad will happen with model after caching and no need to load from scratch again
    # orig_model = get_model(args.base_model, None, args.dtype, args.device_map, trust_remote_code=args.trust_remote_code)
    # if not args.device_map:
    #     orig_model = orig_model.to(device)

    quant_model = load_nncf_quantized_model(args.nncf_ckpt_dir, orig_model, tokenizer)
    generate_chicken(quant_model, tokenizer, "FQ")
    # evaluate_model(quant_model, args)
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
        val_loader=val_dataloader,
        val_hiddens=orig_val_hiddens,
        ckpt_dir=ckpt_dir,
        eval_datasets=eval_datasets
    )

    compute_validation_perplexities(args, quant_model, eval_datasets)

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

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import json
import os
import time
import math
import pickle
import inspect
from dataclasses import MISSING, fields
from contextlib import nullcontext
import argparse

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn

# -----------------------------------------------------------------------------
# Muon optimizer implementation
import torch.distributed as dist

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        weight_decay: Weight decay for regularization.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Single-process fallback: the distributed implementation relies on an initialized
        # default process group for all_gather. For the common 1-GPU/1-process case we
        # run a local Muon update without any distributed collectives.
        if self.world_size == 1 or not dist.is_available() or not dist.is_initialized():
            for group in self.param_groups:
                lr = group["lr"]
                wd = group["weight_decay"]
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]

                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue

                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)

                    if nesterov:
                        g_update = g.add(buf, alpha=momentum)
                    else:
                        g_update = buf

                    g_mat = g_update
                    unflatten_shape = None
                    if g_mat.ndim > 2:
                        unflatten_shape = g_mat.shape
                        g_mat = g_mat.flatten(1)

                    g_ortho = zeropower_via_newtonschulz5(g_mat, steps=ns_steps)
                    if unflatten_shape is not None:
                        g_ortho = g_ortho.view(unflatten_shape)

                    p.mul_(1 - lr * wd)
                    scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5 if p.ndim >= 2 else 1.0
                    p.add_(g_ortho, alpha=-lr * scale)
            return

        for group in self.param_groups:
            update_buffer = group["update_buffer"]
            update_buffer_views = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4: # for the case of conv filters
                        g = g.view(len(g), -1)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

# -----------------------------------------------------------------------------
def to_jsonable(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, torch.Size):
        return list(value)
    return str(value)


def write_json(path, payload):
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2)


def append_jsonl(path, payload):
    with open(path, 'a') as handle:
        handle.write(json.dumps(payload) + '\n')


def config_to_dict(config_obj):
    if hasattr(config_obj, '__dataclass_fields__'):
        return {
            field_name: to_jsonable(getattr(config_obj, field_name))
            for field_name in config_obj.__dataclass_fields__
        }
    return {
        key: to_jsonable(value)
        for key, value in vars(config_obj).items()
        if not key.startswith('_')
    }


def collect_model_internal_snapshot(model_obj, use_baseline_model):
    config_snapshot = config_to_dict(model_obj.config)
    block_summaries = []
    for idx, block in enumerate(getattr(model_obj.transformer, 'h', [])):
        block_summary = {
            'block_index': idx,
            'has_curvature_parameter': hasattr(block, 'c'),
            'has_dynamic_predictor': hasattr(block, 'curvature_predictor'),
        }
        if hasattr(block, 'c') and isinstance(block.c, torch.Tensor):
            block_summary['curvature_shape'] = list(block.c.shape)
            block_summary['curvature_numel'] = block.c.numel()
            block_summary['curvature_mean'] = float(block.c.detach().float().mean().cpu().item())
        block_summaries.append(block_summary)

    has_embedding_curvature = hasattr(model_obj, 'embedding_curvature') and isinstance(model_obj.embedding_curvature, torch.Tensor)
    geometry_regime = 'euclidean_baseline' if use_baseline_model else 'mixed_curvature'
    forward_geometry = {
        'embedding_sum_space': 'euclidean',
        'embedding_expmap_enabled': bool(getattr(model_obj.config, 'use_embedding_curvature', False)),
        'transformer_blocks': 'euclidean' if use_baseline_model else 'hyperbolic_residual_updates',
        'final_norm_and_lm_head': 'euclidean',
    }

    snapshot = {
        'geometry_regime': geometry_regime,
        'model_variant': 'baseline' if use_baseline_model else 'mixed_curvature_transformer',
        'forward_geometry': forward_geometry,
        'config': config_snapshot,
        'parameter_count_total': sum(p.numel() for p in model_obj.parameters()),
        'parameter_count_trainable': sum(p.numel() for p in model_obj.parameters() if p.requires_grad),
        'parameter_count_non_embedding': int(model_obj.get_num_params()) if hasattr(model_obj, 'get_num_params') else None,
        'has_embedding_curvature_parameter': has_embedding_curvature,
        'embedding_curvature_shape': list(model_obj.embedding_curvature.shape) if has_embedding_curvature else None,
        'shared_curvature_shape': list(model_obj.shared_curvature.shape) if hasattr(model_obj, 'shared_curvature') and isinstance(model_obj.shared_curvature, torch.Tensor) else None,
        'block_summaries': block_summaries,
    }
    if has_embedding_curvature:
        snapshot['embedding_curvature_mean'] = float(model_obj.embedding_curvature.detach().float().mean().cpu().item())
    return snapshot


def collect_curvature_logging(raw_model):
    flat_log = {}
    snapshot = {
        'type': 'none',
        'dynamic_curvature': bool(getattr(raw_model.config, 'dynamic_curvature', False)),
        'per_head_curvature': bool(getattr(raw_model.config, 'per_head_curvature', False)),
        'curvature_mode': getattr(raw_model.config, 'curvature_mode', None),
    }

    if hasattr(raw_model.config, 'dynamic_curvature') and raw_model.config.dynamic_curvature:
        all_dynamic_curvature_block_means = []
        block_summaries = []
        flat_log['curvature_type'] = 'dynamic'
        snapshot['type'] = 'dynamic'

        for i, block in enumerate(raw_model.transformer.h):
            if not hasattr(block, 'last_dynamic_c'):
                continue

            dynamic_c_tensor = block.last_dynamic_c
            mean_batch_dynamic_c = dynamic_c_tensor.mean(dim=0).cpu()
            block_summary = {
                'block_index': i,
                'batch_mean_shape': list(mean_batch_dynamic_c.shape),
            }
            current_block_means = []

            if mean_batch_dynamic_c.numel() == raw_model.config.n_head and raw_model.config.per_head_curvature:
                head_means = []
                for h_idx in range(raw_model.config.n_head):
                    head_mean_c = float(mean_batch_dynamic_c[h_idx].item())
                    head_means.append(head_mean_c)
                    flat_log[f'dynamic_curvature/block_{i}/head_{h_idx}_batch_mean'] = head_mean_c
                    current_block_means.append(head_mean_c)
                block_overall_mean = sum(current_block_means) / len(current_block_means) if current_block_means else 0.0
                flat_log[f'dynamic_curvature/block_{i}_overall_batch_mean'] = block_overall_mean
                all_dynamic_curvature_block_means.append(block_overall_mean)
                block_summary['head_means'] = head_means
                block_summary['overall_batch_mean'] = block_overall_mean
            elif mean_batch_dynamic_c.numel() == 1:
                scalar_mean_c = float(mean_batch_dynamic_c.item())
                flat_log[f'dynamic_curvature/block_{i}_batch_mean'] = scalar_mean_c
                all_dynamic_curvature_block_means.append(scalar_mean_c)
                block_summary['scalar_batch_mean'] = scalar_mean_c
            else:
                block_summary['warning'] = f'unexpected_shape:{list(mean_batch_dynamic_c.shape)}'

            block_summaries.append(block_summary)

        snapshot['blocks'] = block_summaries
        if all_dynamic_curvature_block_means:
            snapshot['global_avg_block_means'] = sum(all_dynamic_curvature_block_means) / len(all_dynamic_curvature_block_means)
            snapshot['global_min_block_means'] = min(all_dynamic_curvature_block_means)
            snapshot['global_max_block_means'] = max(all_dynamic_curvature_block_means)
            flat_log['dynamic_curvature/global_avg_block_means'] = snapshot['global_avg_block_means']
            flat_log['dynamic_curvature/global_min_block_means'] = snapshot['global_min_block_means']
            flat_log['dynamic_curvature/global_max_block_means'] = snapshot['global_max_block_means']
        return flat_log, snapshot

    if hasattr(raw_model.config, 'curvature_mode') and raw_model.config.curvature_mode in ['parametric', 'random', 'tied']:
        flat_log['curvature_type'] = 'static'
        snapshot['type'] = 'static'
        all_static_curvature_values = []
        block_summaries = []

        for i, block in enumerate(raw_model.transformer.h):
            if not (hasattr(block, 'c') and isinstance(block.c, nn.Parameter)):
                continue
            block_summary = {'block_index': i}

            if block.c.dim() > 0 and block.c.numel() > 1:
                if block.c.numel() == raw_model.config.n_head:
                    head_values = [float(c_val.item()) for c_val in block.c.detach().cpu()]
                else:
                    head_size = raw_model.config.n_embd // raw_model.config.n_head
                    head_values = [float(block.c.detach().cpu()[h * head_size].item()) for h in range(raw_model.config.n_head)]
                for h, head_val in enumerate(head_values):
                    flat_log[f'curvature/block_{i}/head_{h}'] = head_val
                block_avg = float(block.c.detach().mean().cpu().item())
                flat_log[f'curvature/block_{i}'] = block_avg
                all_static_curvature_values.extend(head_values)
                block_summary['head_values'] = head_values
                block_summary['block_average'] = block_avg
            else:
                c_value = float(block.c.detach().cpu().item())
                flat_log[f'curvature/block_{i}'] = c_value
                all_static_curvature_values.append(c_value)
                block_summary['block_value'] = c_value

            block_summaries.append(block_summary)

        snapshot['blocks'] = block_summaries
        if all_static_curvature_values:
            snapshot['avg'] = sum(all_static_curvature_values) / len(all_static_curvature_values)
            snapshot['min'] = min(all_static_curvature_values)
            snapshot['max'] = max(all_static_curvature_values)
            snapshot['static_per_head_active'] = bool(getattr(raw_model.config, 'per_head_curvature', False))
            flat_log['curvature/avg'] = snapshot['avg']
            flat_log['curvature/min'] = snapshot['min']
            flat_log['curvature/max'] = snapshot['max']
            flat_log['curvature/static_per_head_active'] = snapshot['static_per_head_active']

        if hasattr(raw_model, 'embedding_curvature') and isinstance(raw_model.embedding_curvature, nn.Parameter):
            embedding_curvature = float(raw_model.embedding_curvature.detach().cpu().item())
            snapshot['embedding'] = embedding_curvature
            flat_log['curvature/embedding'] = embedding_curvature
        return flat_log, snapshot

    return flat_log, snapshot


def collect_hyperbolic_debug(raw_model):
    flat_log = {}
    snapshot = {
        'enabled': bool(getattr(raw_model.config, 'hyperbolic_debug', False)),
        'residual_mode': getattr(raw_model.config, 'hyperbolic_residual_mode', None),
        'transport_mode': getattr(raw_model.config, 'hyperbolic_transport_mode', None),
        'project_hyperbolic_points': bool(getattr(raw_model.config, 'project_hyperbolic_points', False)),
    }
    if not snapshot['enabled']:
        return flat_log, snapshot

    block_summaries = []
    global_abs_max = []
    global_nonfinite = []
    global_min_denominator = []

    for i, block in enumerate(raw_model.transformer.h):
        debug_payload = getattr(block, 'last_hyperbolic_debug', None)
        if not debug_payload:
            continue
        block_summary = {'block_index': i}
        for key, value in sorted(debug_payload.items()):
            jsonable = to_jsonable(value)
            block_summary[key] = jsonable
            flat_log[f'hyperbolic_debug/block_{i}/{key}'] = jsonable
            if key.endswith('_abs_max') and isinstance(jsonable, (int, float)):
                global_abs_max.append(float(jsonable))
            if key.endswith('_nonfinite_count') and isinstance(jsonable, (int, float)):
                global_nonfinite.append(float(jsonable))
            if key.endswith('denominator_min_abs') and isinstance(jsonable, (int, float)):
                global_min_denominator.append(float(jsonable))
        block_summaries.append(block_summary)

    snapshot['blocks'] = block_summaries
    if global_abs_max:
        snapshot['global_max_abs_value'] = max(global_abs_max)
    if global_nonfinite:
        snapshot['global_max_nonfinite_count'] = max(global_nonfinite)
    if global_min_denominator:
        snapshot['global_min_denominator_abs'] = min(global_min_denominator)
    return flat_log, snapshot


def collect_named_gradient_stats(named_params):
    total_sq = 0.0
    grad_tensors = 0
    nonfinite_grad_count = 0
    max_abs = 0.0
    first_nonfinite_param = None

    for name, param in named_params:
        grad = param.grad
        if grad is None:
            continue
        grad_tensors += 1
        grad_detached = grad.detach()
        finite_mask = torch.isfinite(grad_detached)
        if finite_mask.any():
            finite_grad = grad_detached[finite_mask].float()
            total_sq += float(torch.sum(finite_grad * finite_grad).cpu().item())
            max_abs = max(max_abs, float(finite_grad.abs().max().cpu().item()))
        if not finite_mask.all():
            nonfinite_grad_count += int((~finite_mask).sum().cpu().item())
            if first_nonfinite_param is None:
                first_nonfinite_param = name

    return {
        'grad_tensors': grad_tensors,
        'grad_l2_norm': math.sqrt(total_sq),
        'grad_abs_max': max_abs,
        'nonfinite_grad_count': nonfinite_grad_count,
        'first_nonfinite_grad_param': first_nonfinite_param,
    }


def collect_named_parameter_stats(named_params):
    nonfinite_param_count = 0
    max_abs = 0.0
    first_nonfinite_param = None

    for name, param in named_params:
        param_detached = param.detach()
        finite_mask = torch.isfinite(param_detached)
        if finite_mask.any():
            max_abs = max(max_abs, float(param_detached[finite_mask].abs().max().cpu().item()))
        if not finite_mask.all():
            nonfinite_param_count += int((~finite_mask).sum().cpu().item())
            if first_nonfinite_param is None:
                first_nonfinite_param = name

    return {
        'parameter_abs_max': max_abs,
        'nonfinite_parameter_count': nonfinite_param_count,
        'first_nonfinite_parameter': first_nonfinite_param,
    }


def unscale_named_grads(named_params, inv_scale):
    for _, param in named_params:
        if param.grad is None:
            continue
        param.grad.detach().mul_(inv_scale)


def build_run_manifest(raw_model, optimizer_metadata, tokens_per_iter, max_iters, out_dir, device, device_type, dtype, compile_enabled, ddp, ddp_world_size, gradient_accumulation_steps, seed_offset):
    return {
        'run': {
            'out_dir': out_dir,
            'dataset': dataset,
            'init_from': init_from,
            'device': device,
            'device_type': device_type,
            'dtype': dtype,
            'compile': compile_enabled,
            'ddp': ddp,
            'ddp_world_size': ddp_world_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'batch_size': batch_size,
            'block_size': block_size,
            'tokens_per_iter': tokens_per_iter,
            'max_iters': max_iters,
            'total_token_budget': tokens_per_iter * max_iters,
            'eval_interval': eval_interval,
            'eval_iters': eval_iters,
            'log_interval': log_interval,
            'always_save_checkpoint': always_save_checkpoint,
            'seed_base': 1337,
            'seed_offset': seed_offset,
        },
        'optimizer': optimizer_metadata,
        'model': collect_model_internal_snapshot(raw_model, use_baseline_model),
        'config': {key: to_jsonable(value) for key, value in config.items()},
        'artifacts': {
            'checkpoint_path': os.path.join(out_dir, 'ckpt.pt'),
            'metrics_jsonl_path': os.path.join(out_dir, 'metrics.jsonl'),
            'eval_history_path': os.path.join(out_dir, 'eval_history.json'),
            'run_manifest_path': os.path.join(out_dir, 'run_manifest.json'),
        },
    }


# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
use_baseline_model = False # whether to use the baseline model from model_baseline.py
curvature_mode = 'random' # 'fixed', 'parametric', 'tied', or 'random' initialization
hyperbolic_debug = False # whether to log lightweight hyperbolic internals during eval
hyperbolic_residual_mode = 'mobius' # 'mobius' or 'euclidean' for residual ablations
hyperbolic_transport_mode = 'logexp' # 'logexp' or 'identity' to bypass tangent transport
project_hyperbolic_points = False # whether to project updates/states back inside the implied ball
hyperbolic_projection_eps = 1e-3
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# muon optimizer
use_muon = False # whether to use Muon optimizer for 2D parameters
muon_lr = 0.02 # learning rate for Muon
muon_momentum = 0.95 # momentum for Muon
muon_nesterov = True # whether to use nesterov momentum in Muon
muon_ns_steps = 5 # number of Newton-Schulz iteration steps
muon_unscale_grads = False # whether to manually unscale Muon gradients when training in float16
optimizer_debug = False # whether to log Muon/AdamW gradient and parameter finite-state summaries
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16'#'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file

# Set output directory based on model type
# Respect explicit run directories from configs/CLI; only auto-name when left at the generic default.
if out_dir == 'out':
    out_dir = 'out_baseline' if use_baseline_model else 'out_hyperbolic'

config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Import appropriate model based on configuration
if use_baseline_model:
    print("Using baseline model from model_baseline.py")
    from model_baseline import GPTConfig, GPT
else:
    print("Using standard model from model.py")
    from model import GPTConfig, GPT

def build_model_args(config_cls):
    """Materialize the selected model config so checkpoints faithfully describe the run."""
    model_args = {}
    for config_field in fields(config_cls):
        if config_field.name in globals():
            model_args[config_field.name] = globals()[config_field.name]
        elif config_field.default is not MISSING:
            model_args[config_field.name] = config_field.default
        elif config_field.default_factory is not MISSING:
            model_args[config_field.name] = config_field.default_factory()
    return model_args

gptconfig_field_names = {config_field.name for config_field in fields(GPTConfig)}

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Checkpoint directory: {out_dir}")
run_manifest_path = os.path.join(out_dir, 'run_manifest.json')
metrics_jsonl_path = os.path.join(out_dir, 'metrics.jsonl')
eval_history_path = os.path.join(out_dir, 'eval_history.json')
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x_np_list = [(data[i:i+block_size]).astype(np.int64) for i in ix]
    y_np_list = [(data[i+1:i+1+block_size]).astype(np.int64) for i in ix]

    # Filter out invalid tokens by replacing them with padding token 0
    for batch_idx in range(batch_size):
        x_np_list[batch_idx] = np.where(x_np_list[batch_idx] > 50257, 0, x_np_list[batch_idx]) #this is because the dataset seems to contain some artifaacts, with indices over 50257. but it's just 20 o them and spaced out quite  systematically, so i'm assuming it's just a mistake in tokenization, and we're still properly tokenizing most tokens.
        y_np_list[batch_idx] = np.where(y_np_list[batch_idx] > 50257, 0, y_np_list[batch_idx])

    x = torch.stack([torch.from_numpy(arr) for arr in x_np_list])
    y = torch.stack([torch.from_numpy(arr) for arr in y_np_list])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = build_model_args(GPTConfig)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    print(f"Using vocab_size = {model_args['vocab_size']} for new model initialization.")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Restore the saved model configuration so resumed runs use the original architecture.
    for k in gptconfig_field_names:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in gptconfig_field_names:
        if hasattr(model.config, k):
            model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer_metadata = {
    'name': 'adamw',
    'use_muon': False,
    'learning_rate': learning_rate,
    'min_lr': min_lr,
    'weight_decay': weight_decay,
    'betas': [beta1, beta2],
}
named_trainable_params = []
named_non_matrix_params = []
named_matrix_params = []
if not use_muon:
    # Use standard AdamW for all parameters
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    named_trainable_params = list(param_dict.items())
    named_non_matrix_params = named_trainable_params
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optimizer_metadata['parameter_split'] = {
        'adamw_decay_tensors': len(decay_params),
        'adamw_decay_parameters': sum(p.numel() for p in decay_params),
        'adamw_nodecay_tensors': len(nodecay_params),
        'adamw_nodecay_parameters': sum(p.numel() for p in nodecay_params),
        'total_trainable_tensors': len(param_dict),
        'total_trainable_parameters': sum(p.numel() for p in param_dict.values()),
    }
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
else:
    # Split parameters into those for AdamW and those for Muon
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    named_trainable_params = list(param_dict.items())
    # Use AdamW for 1D parameters and embeddings
    named_non_matrix_params = [(n, p) for n, p in param_dict.items() if p.dim() < 2 or 'wte' in n or 'wpe' in n or 'lm_head' in n]
    non_matrix_params = [p for _, p in named_non_matrix_params]
    # Use Muon for 2D matrices (except embeddings and lm_head)
    named_matrix_params = [(n, p) for n, p in param_dict.items() if p.dim() >= 2 and 'wte' not in n and 'wpe' not in n and 'lm_head' not in n]
    matrix_params = [p for _, p in named_matrix_params]
    
    # Report parameter split
    num_non_matrix_params = sum(p.numel() for p in non_matrix_params)
    num_matrix_params = sum(p.numel() for p in matrix_params)
    print(f"num parameters for AdamW: {len(non_matrix_params)}, with {num_non_matrix_params:,} parameters")
    print(f"num parameters for Muon: {len(matrix_params)}, with {num_matrix_params:,} parameters")
    optimizer_metadata = {
        'name': 'adamw_plus_muon',
        'use_muon': True,
        'learning_rate': learning_rate,
        'min_lr': min_lr,
        'weight_decay': weight_decay,
        'betas': [beta1, beta2],
        'muon_lr': muon_lr,
        'muon_lr_ratio': (muon_lr / learning_rate) if learning_rate != 0 else None,
        'muon_momentum': muon_momentum,
        'muon_nesterov': muon_nesterov,
        'muon_ns_steps': muon_ns_steps,
        'parameter_split': {
            'adamw_tensors': len(non_matrix_params),
            'adamw_parameters': num_non_matrix_params,
            'muon_tensors': len(matrix_params),
            'muon_parameters': num_matrix_params,
            'total_trainable_tensors': len(param_dict),
            'total_trainable_parameters': sum(p.numel() for p in param_dict.values()),
        },
    }
    
    # Create AdamW optimizer for non-matrix parameters
    # Create optim groups with weight decay for parameters from AdamW
    decay_params = [p for p in non_matrix_params if p.dim() >= 2]
    nodecay_params = [p for p in non_matrix_params if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Create AdamW optimizer with fused implementation if available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    adamw_optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)
    print(f"using fused AdamW: {use_fused}")
    
    # Create Muon optimizer for matrix parameters
    ddp_rank = int(os.environ.get('RANK', -1))
    ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
    if ddp:
        muon_optimizer = Muon(
            matrix_params, 
            lr=muon_lr, 
            weight_decay=weight_decay,
            momentum=muon_momentum, 
            nesterov=muon_nesterov, 
            ns_steps=muon_ns_steps,
            rank=ddp_rank, 
            world_size=ddp_world_size
        )
    else:
        muon_optimizer = Muon(
            matrix_params, 
            lr=muon_lr, 
            weight_decay=weight_decay,
            momentum=muon_momentum, 
            nesterov=muon_nesterov, 
            ns_steps=muon_ns_steps,
            rank=0, 
            world_size=1
        )
    
    # Combine optimizers
    optimizer = [adamw_optimizer, muon_optimizer]

if init_from == 'resume':
    if not use_muon or not isinstance(optimizer, list):
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        # For the case of resuming with Muon when the checkpoint used a single optimizer
        # This is a simplification and may need more complex handling in a real scenario
        print("Resuming with Muon from a non-Muon checkpoint - optimizer state will be reset")
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    #wandb.watch(model, log="all", log_freq=eval_interval) # log gradients and parameters

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
eval_history = []
if master_process:
    write_json(
        run_manifest_path,
        build_run_manifest(
            raw_model=raw_model,
            optimizer_metadata=optimizer_metadata,
            tokens_per_iter=tokens_per_iter,
            max_iters=max_iters,
            out_dir=out_dir,
            device=device,
            device_type=device_type,
            dtype=dtype,
            compile_enabled=compile,
            ddp=ddp,
            ddp_world_size=ddp_world_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            seed_offset=seed_offset,
        ),
    )
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    if use_muon and isinstance(optimizer, list):
        # Apply learning rate to both optimizers
        for param_group in optimizer[0].param_groups:
            param_group['lr'] = lr
        # For Muon, scale the learning rate appropriately
        muon_lr_scaled = lr * (muon_lr / learning_rate)
        for param_group in optimizer[1].param_groups:
            param_group['lr'] = muon_lr_scaled
    else:
        # Standard optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        curvature_log_dict, curvature_snapshot = collect_curvature_logging(raw_model)
        hyperbolic_log_dict, hyperbolic_snapshot = collect_hyperbolic_debug(raw_model)
        eval_record = {
            'event': 'eval',
            'iter': iter_num,
            'train_loss': float(losses['train']),
            'val_loss': float(losses['val']),
            'lr': float(lr),
            'mfu': None if running_mfu < 0 else float(running_mfu * 100),
            'curvature': curvature_snapshot,
            'hyperbolic_debug': hyperbolic_snapshot,
        }
        eval_history.append(eval_record)
        append_jsonl(metrics_jsonl_path, eval_record)
        write_json(eval_history_path, eval_history)
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            log_dict.update(curvature_log_dict)
            log_dict.update(hyperbolic_log_dict)
            wandb.log(log_dict)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict() if not use_muon or not isinstance(optimizer, list) else None,
                    'optimizers': [opt.state_dict() for opt in optimizer] if use_muon and isinstance(optimizer, list) else None,
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                model_type = "baseline" if use_baseline_model else "hyperbolic"
                print(f"saving {model_type} model checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    optimizer_step_debug = None
    if use_muon and isinstance(optimizer, list):
        need_adamw_unscale = grad_clip != 0.0 or muon_unscale_grads or optimizer_debug
        if need_adamw_unscale:
            scaler.unscale_(optimizer[0])
        if muon_unscale_grads and dtype == 'float16':
            inv_scale = 1.0 / scaler.get_scale()
            unscale_named_grads(named_matrix_params, inv_scale)
        if optimizer_debug:
            optimizer_step_debug = {
                'grad_scaler_scale': float(scaler.get_scale()) if dtype == 'float16' else 1.0,
                'muon_unscale_grads': bool(muon_unscale_grads),
                'before_step': {
                    'adamw': collect_named_gradient_stats(named_non_matrix_params),
                    'muon': collect_named_gradient_stats(named_matrix_params),
                    'parameters': collect_named_parameter_stats(named_trainable_params),
                },
            }
    # clip the gradient
    if grad_clip != 0.0:
        if not use_muon or not isinstance(optimizer, list):
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    if use_muon and isinstance(optimizer, list):
        scaler.step(optimizer[0])
        if optimizer_step_debug is not None:
            optimizer_step_debug['after_adamw'] = collect_named_parameter_stats(named_trainable_params)
        optimizer[1].step()
        if optimizer_step_debug is not None:
            optimizer_step_debug['after_muon'] = collect_named_parameter_stats(named_trainable_params)
    else:
        scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    if use_muon and isinstance(optimizer, list):
        optimizer[0].zero_grad(set_to_none=True)
        optimizer[1].zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        append_jsonl(
            metrics_jsonl_path,
            {
                'event': 'train_step',
                'iter': iter_num,
                'loss': float(lossf),
                'lr': float(lr),
                'time_ms': float(dt * 1000),
                'mfu': None if running_mfu < 0 else float(running_mfu * 100),
                'optimizer_debug': optimizer_step_debug,
            },
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

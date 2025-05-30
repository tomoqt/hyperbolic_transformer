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

import os
import time
import math
import pickle
import inspect
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
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
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
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
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
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
if not use_muon:
    # Use standard AdamW for all parameters
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
else:
    # Split parameters into those for AdamW and those for Muon
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Use AdamW for 1D parameters and embeddings
    non_matrix_params = [p for n, p in param_dict.items() if p.dim() < 2 or 'wte' in n or 'wpe' in n or 'lm_head' in n]
    # Use Muon for 2D matrices (except embeddings and lm_head)
    matrix_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'wte' not in n and 'wpe' not in n and 'lm_head' not in n]
    
    # Report parameter split
    num_non_matrix_params = sum(p.numel() for p in non_matrix_params)
    num_matrix_params = sum(p.numel() for p in matrix_params)
    print(f"num parameters for AdamW: {len(non_matrix_params)}, with {num_non_matrix_params:,} parameters")
    print(f"num parameters for Muon: {len(matrix_params)}, with {num_matrix_params:,} parameters")
    
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
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            
            # Log curvature values to wandb
            if hasattr(raw_model.config, 'dynamic_curvature') and raw_model.config.dynamic_curvature:
                all_dynamic_curvature_block_means = []
                log_dict['curvature_type'] = 'dynamic'

                for i, block in enumerate(raw_model.transformer.h):
                    if hasattr(block, 'last_dynamic_c'):
                        # dynamic_c_tensor has shape (B, n_head) or (B, 1)
                        # B is batch_size used in estimate_loss, which is the main script's batch_size
                        dynamic_c_tensor = block.last_dynamic_c 
                        
                        # Calculate mean over the batch dimension. Result shape: (n_head,) or (1,)
                        mean_batch_dynamic_c = dynamic_c_tensor.mean(dim=0).cpu()
                        
                        current_block_means = []
                        if mean_batch_dynamic_c.numel() == raw_model.config.n_head and raw_model.config.per_head_curvature:
                            # Per-head dynamic curvature prediction
                            for h_idx in range(raw_model.config.n_head):
                                head_mean_c = mean_batch_dynamic_c[h_idx].item()
                                log_dict[f'dynamic_curvature/block_{i}/head_{h_idx}_batch_mean'] = head_mean_c
                                current_block_means.append(head_mean_c)
                            block_overall_mean = sum(current_block_means) / len(current_block_means) if current_block_means else 0.0
                            log_dict[f'dynamic_curvature/block_{i}_overall_batch_mean'] = block_overall_mean
                            all_dynamic_curvature_block_means.append(block_overall_mean)
                        elif mean_batch_dynamic_c.numel() == 1:
                            # Scalar dynamic curvature prediction
                            scalar_mean_c = mean_batch_dynamic_c.item()
                            log_dict[f'dynamic_curvature/block_{i}_batch_mean'] = scalar_mean_c
                            all_dynamic_curvature_block_means.append(scalar_mean_c)
                        else:
                            print(f"Warning: Unexpected shape for mean_batch_dynamic_c in block {i}: {mean_batch_dynamic_c.shape}")
                
                if all_dynamic_curvature_block_means:
                    log_dict['dynamic_curvature/global_avg_block_means'] = sum(all_dynamic_curvature_block_means) / len(all_dynamic_curvature_block_means)
                    log_dict['dynamic_curvature/global_min_block_means'] = min(all_dynamic_curvature_block_means)
                    log_dict['dynamic_curvature/global_max_block_means'] = max(all_dynamic_curvature_block_means)
                
                # It's good practice to remove the stored tensors after use if they are not needed elsewhere before the next eval
                # to ensure fresh values and free memory, though detach() helps with graph issues.
                # However, estimate_loss runs multiple times, so this must be done carefully or not at all if it interferes.
                # For now, let's assume they get overwritten or cleared correctly by the model logic.

            # Log static/base curvatures if dynamic curvature is NOT active
            elif hasattr(raw_model.config, 'curvature_mode') and raw_model.config.curvature_mode in ['parametric', 'random','tied']:
                log_dict['curvature_type'] = 'static'
                all_static_curvature_values = []
                for i, block in enumerate(raw_model.transformer.h):
                    if hasattr(block, 'c') and isinstance(block.c, nn.Parameter):
                        # Check if using per-head curvature
                        if block.c.dim() > 0 and block.c.numel() > 1:
                            # Log per-head curvature values
                            if block.c.numel() == raw_model.config.n_head:
                                # Original per-head version (one value per head)
                                for h, c_val in enumerate(block.c.detach().cpu()):
                                    log_dict[f'curvature/block_{i}/head_{h}'] = c_val.item()
                                    all_static_curvature_values.append(c_val.item())
                                # Also log block average
                                block_avg = block.c.detach().mean().cpu().item()
                                log_dict[f'curvature/block_{i}'] = block_avg
                            else:
                                # Expanded version where each head's value is repeated
                                # Sample just the first value for each head to avoid too many logs
                                head_size = raw_model.config.n_embd // raw_model.config.n_head
                                for h in range(raw_model.config.n_head):
                                    head_val = block.c.detach().cpu()[h * head_size].item()
                                    log_dict[f'curvature/block_{i}/head_{h}'] = head_val
                                    all_static_curvature_values.append(head_val)
                                # Also log block average
                                block_avg = block.c.detach().mean().cpu().item()
                                log_dict[f'curvature/block_{i}'] = block_avg
                        else:
                            # Original single curvature per block
                            c_value = block.c.detach().cpu().item()
                            log_dict[f'curvature/block_{i}'] = c_value
                            all_static_curvature_values.append(c_value)
                
                # Add curvature statistics
                if all_static_curvature_values:
                    log_dict['curvature/avg'] = sum(all_static_curvature_values) / len(all_static_curvature_values)
                    log_dict['curvature/min'] = min(all_static_curvature_values)
                    log_dict['curvature/max'] = max(all_static_curvature_values)
                    
                    # Add a flag to indicate per-head curvature is being used for static
                    log_dict['curvature/static_per_head_active'] = hasattr(raw_model.config, 'per_head_curvature') and raw_model.config.per_head_curvature
                
                # Add embedding curvature if it exists (this is always static if present)
                if hasattr(raw_model, 'embedding_curvature') and isinstance(raw_model.embedding_curvature, nn.Parameter):
                    embedding_curvature = raw_model.embedding_curvature.detach().cpu().item()
                    log_dict['curvature/embedding'] = embedding_curvature
                    all_static_curvature_values.append(embedding_curvature)
                    # Update statistics to include embedding curvature
                    log_dict['curvature/avg'] = sum(all_static_curvature_values) / len(all_static_curvature_values)
                    log_dict['curvature/min'] = min(all_static_curvature_values)
                    log_dict['curvature/max'] = max(all_static_curvature_values)
            
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
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer if not use_muon or not isinstance(optimizer, list) else optimizer[0])
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    if use_muon and isinstance(optimizer, list):
        scaler.step(optimizer[0])
        optimizer[1].step()
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
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
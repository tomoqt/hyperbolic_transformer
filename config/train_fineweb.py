# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'finewebT'
wandb_run_name='looped'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 1
block_size = 512
gradient_accumulation_steps =  8
dataset = 'fineweb'
# this makes total number of tokens be 300B
max_iters = 60000
lr_decay_iters = 60000
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
# eval stuff
eval_interval = 100
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

use_muon = False
muon_lr = 1e-3
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5

# Looping configurations
max_loops = 30
loop_groups = [[2, 3], [4]] # Example: loop layers 2 and 3 (0-indexed)
loop_counts = None       # Number of loops for each group in loop_groups. None means use max_loops.
loop_noise_scale = 0.0
concatenate_initial_representation = True
loops_representation = False # For debugging/analysis
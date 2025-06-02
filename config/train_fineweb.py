

wandb_log = True
wandb_project = 'fineweb-looped-small'
wandb_run_name='looped-small'

batch_size = 12
block_size = 512
gradient_accumulation_steps =  1
dataset = 'fineweb'
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
max_loops = 15
loop_groups = [[2],[3],[4]] # Example: loop layers 2 and 3 (0-indexed)
loop_noise_scale = 1.0
concatenate_initial_representation = True
loops_representation = False # For debugging/analysis
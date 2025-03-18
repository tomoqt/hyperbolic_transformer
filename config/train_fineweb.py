out_dir = 'out-fineweb'
eval_interval = 100 
eval_iters = 20
log_interval = 1

always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'hyperbolic_gpt_fineweb'
wandb_run_name = 'mini-gpt'

dataset = 'fineweb'
gradient_accumulation_steps = 1
# Muon optimizer config
use_muon = False # whether to use Muon optimizer for matrix parameters
muon_lr = 0.0002 # learning rate for Muon optimizer
muon_momentum = 0.95 # momentum for Muon optimizer
muon_nesterov = True # whether to use Nesterov momentum for Muon
muon_ns_steps = 5 # number of Newton-Schulz steps for Muon
batch_size = 4
block_size = 256 

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0
curvature = 1.0
learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
compile = False


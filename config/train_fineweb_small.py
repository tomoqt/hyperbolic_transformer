# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'fineweb_mixed_curvature'
wandb_run_name='small_fineweb'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 1
block_size = 512
gradient_accumulation_steps = 8
dataset = 'fineweb'

# this makes total number of tokens be 300B
max_iters = 60000
lr_decay_iters = 60000

# eval stuff
eval_interval = 100
eval_iters = 20
log_interval = 10

curvature = 1.0

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
use_muon = False
muon_lr = 1e-2
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5
learning_rate = 1e-4
max_iters = 60000
lr_decay_iters = 60000
min_lr = 1e-5 
beta2 = 0.99 

warmup_iters = 100 # not super necessary potentially
use_embedding_curvature = False
use_baseline_model = False
dynamic_curvature = True
per_head_curvature = False
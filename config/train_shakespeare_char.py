# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'looping-out-shakespeare-char'
eval_interval = 10 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'looped-shakespeare-char'
wandb_run_name = 'looped_shakespeare-char'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 8
block_size = 256 # context of up to 256 previous characters
curvature = 1.0
# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
use_muon = False
muon_lr = 1e-2
muon_momentum = 0.95
muon_nesterov = True
muon_ns_steps = 5
learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
map_back_after_attention = False
use_embedding_curvature = False
# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
# Looping configurations
max_loops = 20
loop_groups = [[2],[3],[4]] # Example: loop layers 2 and 3 (0-indexed)
loop_noise_scale = 1.0
concatenate_initial_representation = True
loops_representation = False # For debugging/analysis
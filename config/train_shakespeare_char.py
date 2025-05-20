# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'Hyperbolic-out-shakespeare-char'
eval_interval = 10 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 1 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'hyperbolic-shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
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
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
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

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
muon_lr = 1e-3
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
# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
curvature_mode 'tied' # 'fixed', 'parametric', 'tied', or any other value for random init
curvature 0.0 # Fixed curvature value when curvature_mode is 'fixed'
curvature_initialization: list = field(default_factory=list) # List of initial curvature values for parametric mode (one per layer when provided)
map_back_after_attention: bool = False # whether to map back to hyperbolic space after attention or after the MLP
per_head_curvature: bool = True # whether to use a different curvature for each head
hyperbolic_embedding: bool = False # whether to use hyperbolic embeddings
embedding_curvature_mode: str = 'random' # 'fixed', 'parametric', 'tied', or any other value for random init
embedding_curvature: float = 1.0 # Fixed curvature value when embedding_curvature_mode is 'fixed'
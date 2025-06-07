"""
Full definition of a GPT Language Model with higher-order attention.
Adds one "mixed" block (order-2 followed by order-3 attention) every 4
regular blocks.  Drop-in replacement for the original minGPT file.
"""

import math, inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------
#  LayerNorm with optional bias
# ---------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

# ---------------------------------------------------------------------
#  Rotary + RMSNorm helpers (optional, unused here but left for completeness)
# ---------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)

# ------------------------------------------------------------------
#  Higher-order attention  (causal, top-k, order ≥ 2)
# ------------------------------------------------------------------
class HigherOrderAttention(nn.Module):
    letters = "ijklmnopqrstuvwxyz"

    def __init__(self, order: int, n_head: int, embed_dim: int,
                 head_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        assert order >= 2
        self.order, self.n_head, self.head_dim = order, n_head, head_dim
        self.scale = head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.q_proj  = nn.Linear(embed_dim, n_head * head_dim, bias=False)
        self.k_projs = nn.ModuleList([nn.Linear(embed_dim, n_head * head_dim, bias=False)
                                      for _ in range(order - 1)])
        self.v_projs = nn.ModuleList([nn.Linear(embed_dim, n_head * head_dim, bias=False)
                                      for _ in range(order - 1)])
        self.out_proj = nn.Linear(n_head * head_dim, embed_dim, bias=False)

    # ---------------- helper: gather along time --------------------
    @staticmethod
    def _gather_time(tensor, idx):
        """
        tensor : (B, H, T, D)
        idx    : (B, H, T, k)   indices < T   (no future positions)
        return : (B, H, T, k, D)
        """
        B, H, T, D = tensor.shape
        k = idx.size(-1)
        # make both tensors 5-D so torch.gather ranks match
        tensor_5d = tensor.unsqueeze(3).expand(-1, -1, -1, k, -1)        # (B,H,T,k,D)
        idx_5d    = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)          # (B,H,T,k,D)
        return torch.gather(tensor_5d, 2, idx_5d)                        # dim=2 is T

    # ---------------- split heads ----------------------------------
    @staticmethod
    def _split_head(x, n_head, head_dim):
        B, T, _ = x.shape
        return x.view(B, T, n_head, head_dim).transpose(1, 2).contiguous()  # (B,H,T,D)

    # ---------------- forward --------------------------------------
    def forward(self, x):
        B, T, _ = x.shape
        k_keep = max(1, math.ceil(T ** (2.0 / self.order)))
        dev = x.device
        t_arange = torch.arange(T, device=dev)

        # 1) project Q,K,V as before
        q = self._split_head(self.q_proj(x), self.n_head, self.head_dim)
        Ks = [self._split_head(kp(x), self.n_head, self.head_dim) for kp in self.k_projs]
        Vs = [self._split_head(vp(x), self.n_head, self.head_dim) for vp in self.v_projs]

        gathered_K, gathered_V, letters = [], [], []
        for r, (K_r, V_r) in enumerate(zip(Ks, Vs)):
            # 2) compute raw logits and mask future positions
            logits = torch.einsum("b h t d, b h s d -> b h t s", q, K_r) * self.scale
            causal_mask = (t_arange[None,None,:,None] < t_arange[None,None,None,:])  # True where s>t
            logits = logits.masked_fill(causal_mask, float("-inf"))

            # 3) dynamic per-time top-k
            #    build an empty tensor for indices of shape (B,H,T,k_keep)
            topk_idx = torch.zeros(B, self.n_head, T, k_keep, dtype=torch.long, device=dev)
            for t in range(T):
                # only consider keys 0..t
                valid_slice = logits[:, :, t, :t+1]                # (B,H,t+1)
                this_k = min(k_keep, t+1)
                _, idx_t = valid_slice.topk(this_k, dim=-1)        # (B,H,this_k)
                if this_k < k_keep:
                    # pad the rest with the last valid index (or zero)
                    pad = idx_t[:, :, -1:].expand(-1, -1, k_keep - this_k)
                    idx_t = torch.cat([idx_t, pad], dim=-1)       # (B,H,k_keep)
                topk_idx[:, :, t, :] = idx_t

            # store for debug / forward-leak test
            self._last_topk = topk_idx

            # 4) gather the actual K/V
            gathered_K.append(self._gather_time(K_r, topk_idx))
            gathered_V.append(self._gather_time(V_r, topk_idx))
            letters.append(self.letters[r+1])

        # 5) compute higher-order logits & values exactly as before
        einsum_in  = ["b h i d"] + [f"b h i {ltr} d" for ltr in letters]
        einsum_out = "b h i " + "".join(letters)
        A = torch.einsum(", ".join(einsum_in) + " -> " + einsum_out,
                         q, *gathered_K) * self.scale
        alpha = self.dropout(torch.softmax(A, dim=-1))

        einsum_in  = ["b h i " + "".join(letters)] + \
                     [f"b h i {ltr} d" for ltr in letters]
        out = torch.einsum(", ".join(einsum_in) + " -> b h i d",
                           alpha, *gathered_V)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.out_proj(out)

# ---------------------------------------------------------------------
#  Vanilla causal self-attention (order-2)
# ---------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.dropout  = config.dropout
        self.flash    = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y   = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

# ---------------------------------------------------------------------
#  Feed-forward
# ---------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

# ---------------------------------------------------------------------
#  Standard Transformer block  (order-2 only)
# ---------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ---------------------------------------------------------------------
#  Mixed block: order-2 followed immediately by order-3
# ---------------------------------------------------------------------
class MixedBlock(nn.Module):
    """Used every 4th position in the stack."""
    def __init__(self, config):
        super().__init__()
        self.ln_1  = LayerNorm(config.n_embd, bias=config.bias)
        self.attn2 = CausalSelfAttention(config)                             # order-2
        self.ln_hoa = LayerNorm(config.n_embd, bias=config.bias)
        self.attn3  = HigherOrderAttention(order=3,
                                           n_head=config.n_head,
                                           embed_dim=config.n_embd,
                                           head_dim=config.n_embd // config.n_head,
                                           dropout=config.dropout)           # order-3
        self.gate = nn.Linear(config.n_embd, 1)
        self.top_k = getattr(config, 'hoa_top_k', None)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn2(self.ln_1(x))
        hoa_in = self.ln_hoa(x)
        hoa_out = self.attn3(hoa_in)

        scores = self.gate(hoa_in).squeeze(-1)
        k = scores.size(1) if self.top_k is None else min(self.top_k, scores.size(1))
        topk_idx = scores.topk(k, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=hoa_out.dtype)
        mask.scatter_(1, topk_idx, 1.0)
        mask = mask.unsqueeze(-1)

        x = x + hoa_out * mask
        x = x + self.mlp(self.ln_2(x))
        return x

# ---------------------------------------------------------------------
#  GPT configuration
# ---------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size : int  = 1024
    vocab_size : int  = 50304
    n_layer    : int  = 12
    n_head     : int  = 12
    n_embd     : int  = 768
    dropout    : float = 0.0
    bias       : bool  = True
    hoa_top_k  : int   = 8

# ---------------------------------------------------------------------
#  GPT model
# ---------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([
                    MixedBlock(config) if (i % 4 == 3) else Block(config)
                    for i in range(config.n_layer)
                ]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    # --------------------- utilities ---------------------------------
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -------------------- forward ------------------------------------
    def forward(self, idx, targets=None):
        device = idx.device
        b, t   = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
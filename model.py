"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperbolic geometry utility functions
def clamp_curvature(c):
    if isinstance(c, torch.Tensor):
        return torch.clamp(c, min=1e-4, max=1.0)
    else:
        return max(1e-4, min(c, 1.0))

def mobius_addition(x, y, c, n_head=None):
    """Mobius addition in hyperbolic space with curvature c"""
    c_clamped = clamp_curvature(c)

    # x.shape[0] > 0 to prevent issues with dummy tensors during init - B generally > 0 in practice
    B_dim = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim > 0 else 0

    use_per_head_path = False
    c_r = None # Broadcastable curvature for per-head operations

    if n_head is not None and isinstance(c_clamped, torch.Tensor) and B_dim > 0:
        if c_clamped.ndim == 1 and c_clamped.shape[0] == n_head:
            # Case 1: Static per-head curvature, c_clamped shape is (n_head,)
            c_r = c_clamped.view(1, 1, n_head, 1)
            use_per_head_path = True
        elif c_clamped.ndim == 2 and c_clamped.shape[0] == B_dim and c_clamped.shape[1] == n_head:
            # Case 2: Dynamic per-head curvature, c_clamped shape is (B, n_head)
            c_r = c_clamped.view(B_dim, 1, n_head, 1) # Reshape for broadcasting with (B, T, n_head, hs)
            use_per_head_path = True

    if use_per_head_path:
        # Per-head curvature logic
        B, T, C_embed = x.shape # B will match B_dim
        hs = C_embed // n_head
        
        x_r = x.view(B, T, n_head, hs)
        y_r = y.view(B, T, n_head, hs)
        # c_r is already prepared: (1,1,n_head,1) or (B,1,n_head,1)

        # Compute norms and inner product per head
        x_norm_sq_r = torch.sum(x_r * x_r, dim=-1, keepdim=True) # (B, T, n_head, 1)
        y_norm_sq_r = torch.sum(y_r * y_r, dim=-1, keepdim=True) # (B, T, n_head, 1)
        inner_product_r = torch.sum(x_r * y_r, dim=-1, keepdim=True) # (B, T, n_head, 1)

        # Mobius addition formula components
        numerator_r = (1 + 2 * c_r * inner_product_r + c_r * y_norm_sq_r) * x_r + \
                      (1 - c_r * x_norm_sq_r) * y_r
        denominator_r = 1 + 2 * c_r * inner_product_r + c_r**2 * x_norm_sq_r * y_norm_sq_r
        
        result_r = numerator_r / (denominator_r + 1e-9) # Add epsilon to denominator
        return result_r.view(B, T, C_embed) # Reshape back
    else:
        # Original logic for scalar c or c broadcast over embedding dim
        x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
        y_norm_sq = torch.sum(y * y, dim=-1, keepdim=True)
        inner_product = torch.sum(x * y, dim=-1, keepdim=True)
        
        numerator = (1 + 2*c_clamped * inner_product + c_clamped * y_norm_sq) * x + \
                    (1 - c_clamped * x_norm_sq) * y
        denominator = 1 + 2*c_clamped * inner_product + (c_clamped ** 2) * (x_norm_sq) * (y_norm_sq)
        return numerator / (denominator + 1e-9) # Add epsilon to denominator

def scaling_factor(x, c, n_head=None):
    """Compute scaling factor for hyperbolic space with curvature c"""
    c_clamped = clamp_curvature(c)
    
    B_dim = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim > 0 else 0

    # Determine if per-head path should be used and prepare c_r (broadcastable c)
    use_per_head_path = False
    c_r_for_scaling = None

    if n_head is not None and isinstance(c_clamped, torch.Tensor) and B_dim > 0:
        if c_clamped.ndim == 1 and c_clamped.shape[0] == n_head:
            # Static per-head: c_clamped is (n_head,)
            c_r_for_scaling = c_clamped.view(1, 1, n_head, 1)
            use_per_head_path = True
        elif c_clamped.ndim == 2 and c_clamped.shape[0] == B_dim and c_clamped.shape[1] == n_head:
            # Dynamic per-head: c_clamped is (B, n_head)
            c_r_for_scaling = c_clamped.view(B_dim, 1, n_head, 1)
            use_per_head_path = True

    if use_per_head_path:
        B, T, C_embed = x.shape
        hs = C_embed // n_head
        x_r = x.view(B, T, n_head, hs)
        # c_r_for_scaling is (1,1,n_head,1) or (B,1,n_head,1)

        x_norm_sq_r = torch.sum(x_r * x_r, dim=-1, keepdim=True) # (B, T, n_head, 1)
        # Returns (B, T, n_head, 1)
        return 2 / (1 + c_r_for_scaling * x_norm_sq_r + 1e-9) 
    else:
        # Original logic for scalar c, or c broadcast over embedding dim, or dynamic scalar c (B,1)
        x_norm_sq = torch.sum(x*x, dim=-1, keepdim=True) # (B,T,1)
        # c_clamped can be scalar, (B,1), or (n_embd,)
        # Result can be (B,T,1) or (B,T,n_embd) if c_clamped was (n_embd,)
        return 2 / (1 + c_clamped * x_norm_sq + 1e-9)

def expmap(x, v, c, n_head=None):
    """Exponential map from tangent space to hyperbolic space with curvature c"""
    c_clamped = clamp_curvature(c)
    B_dim = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim > 0 else 0

    use_per_head_path = False
    c_br_for_exp = None # Broadcastable c, (1,1,n_head,1) or (B,1,n_head,1)

    if n_head is not None and isinstance(c_clamped, torch.Tensor) and B_dim > 0:
        if c_clamped.ndim == 1 and c_clamped.shape[0] == n_head:
            c_br_for_exp = c_clamped.view(1, 1, n_head, 1)
            use_per_head_path = True
        elif c_clamped.ndim == 2 and c_clamped.shape[0] == B_dim and c_clamped.shape[1] == n_head:
            c_br_for_exp = c_clamped.view(B_dim, 1, n_head, 1)
            use_per_head_path = True

    if use_per_head_path:
        B, T, C_embed = x.shape # B will match B_dim
        hs = C_embed // n_head
        
        # Pass c_clamped (original per-head c) to mobius_addition and scaling_factor
        # as they have their own logic to create c_r / c_br from it.
        sf_x_r = scaling_factor(x, c_clamped, n_head=n_head) # sf_x_r is (B,T,n_head,1)

        v_r = v.view(B, T, n_head, hs)
        v_norm_sq_r = torch.sum(v_r * v_r, dim=-1, keepdim=True) # (B,T,n_head,1)
        v_norm_r = torch.sqrt(v_norm_sq_r + 1e-9) # (B,T,n_head,1), epsilon for stability

        # c_br_for_exp is the correctly shaped (1,1,n_head,1) or (B,1,n_head,1) curvature
        tanh_arg_val = torch.abs(c_br_for_exp * sf_x_r * v_norm_sq_r / 2) 

        sqrt_c_br = torch.sqrt(torch.abs(c_br_for_exp) + 1e-9) # abs for safety, though c should be >0
        sqrt_tanh_arg_val = torch.sqrt(tanh_arg_val + 1e-9) # Epsilon for stability
        
        second_term_coeff = (1 / (sqrt_c_br + 1e-9)) * torch.tanh(sqrt_tanh_arg_val) # Epsilon for sqrt_c_br division
        second_term_r = second_term_coeff * (v_r / (v_norm_r + 1e-9)) # (B, T, n_head, hs)
        
        # Pass original c_clamped and n_head to mobius_addition
        return mobius_addition(x, second_term_r.reshape(B, T, C_embed), c_clamped, n_head=n_head)
    else:
        # Original non-per-head logic (handles scalar c, (B,1) c, or (n_embd,) c)
        sf_x = scaling_factor(x, c_clamped, n_head=None) # (B,T,1) or (B,T,n_embd)
        v_norm_sq = torch.sum(v*v, dim=-1, keepdim=True) # (B,T,1)
        v_norm = torch.sqrt(v_norm_sq + 1e-9) # (B,T,1)

        # Ensure c_clamped is positive for sqrt, clamp_curvature handles this.
        c_clamped_sqrt = torch.sqrt(c_clamped + 1e-9) 
        # tanh_arg base should be positive. sf_x can be (B,T,n_embd).
        # (c_clamped * sf_x * v_norm_sq / 2)
        tanh_arg_base = c_clamped * sf_x * v_norm_sq / 2.0
        tanh_input_sqrt = torch.sqrt(torch.abs(tanh_arg_base) + 1e-9) # abs and epsilon for robustness

        term_coeff = (1 / (c_clamped_sqrt + 1e-9)) * torch.tanh(tanh_input_sqrt)
        second_term_direction = v / (v_norm + 1e-9)
        second_term = term_coeff * second_term_direction
        return mobius_addition(x, second_term, c_clamped, n_head=None)

def logmap(x, u, c, n_head=None):
    """Logarithmic map from hyperbolic space to tangent space with curvature c"""
    c_clamped = clamp_curvature(c)
    B_dim = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim > 0 else 0

    use_per_head_path = False
    c_br_for_log = None # Broadcastable c for logmap, (1,1,n_head,1) or (B,1,n_head,1)

    if n_head is not None and isinstance(c_clamped, torch.Tensor) and B_dim > 0:
        if c_clamped.ndim == 1 and c_clamped.shape[0] == n_head:
            c_br_for_log = c_clamped.view(1, 1, n_head, 1)
            use_per_head_path = True
        elif c_clamped.ndim == 2 and c_clamped.shape[0] == B_dim and c_clamped.shape[1] == n_head:
            c_br_for_log = c_clamped.view(B_dim, 1, n_head, 1)
            use_per_head_path = True

    if use_per_head_path:
        B, T, C_embed = x.shape # B matches B_dim
        hs = C_embed // n_head

        # Pass original c_clamped and n_head to mobius_addition and scaling_factor
        mob_add_result = mobius_addition(-x, u, c_clamped, n_head=n_head) # (B,T,C_embed)
        mob_add_result_r = mob_add_result.view(B, T, n_head, hs) # (B,T,n_head,hs)

        sf_x_r = scaling_factor(x, c_clamped, n_head=n_head) # (B,T,n_head,1)

        addition_norm_sq_r = torch.sum(mob_add_result_r * mob_add_result_r, dim=-1, keepdim=True) # (B,T,n_head,1)
        addition_norm_r = torch.sqrt(addition_norm_sq_r + 1e-9) # (B,T,n_head,1)

        # c_br_for_log is the correctly shaped (1,1,n_head,1) or (B,1,n_head,1) curvature
        sqrt_c_br = torch.sqrt(torch.abs(c_br_for_log) + 1e-9) # abs for safety
        constant_factor = 2 / (sf_x_r * sqrt_c_br + 1e-9) # (B,T,n_head,1)
        direction_factor_r = mob_add_result_r / (addition_norm_r + 1e-9) # (B,T,n_head,hs)
        
        arctanh_arg_r = sqrt_c_br * addition_norm_r # (B,T,n_head,1)
        arctanh_arg_clamped_r = torch.clamp(arctanh_arg_r, min=-0.9999, max=0.9999) # clamp before arctanh
        
        result_r = constant_factor * torch.arctanh(arctanh_arg_clamped_r) * direction_factor_r
        return result_r.reshape(B, T, C_embed)
    else:
        # Original non-per-head logic (handles scalar c, (B,1) c, or (n_embd,) c)
        mob_addition_val = mobius_addition(-x, u, c_clamped, n_head=None) # (B,T,n_embd)
        sf_x = scaling_factor(x, c_clamped, n_head=None) # (B,T,1) or (B,T,n_embd)

        addition_norm_sq = torch.sum(mob_addition_val * mob_addition_val, dim=-1, keepdim=True) # (B,T,1)
        addition_norm = torch.sqrt(addition_norm_sq + 1e-9) # (B,T,1)
        
        c_clamped_sqrt = torch.sqrt(c_clamped + 1e-9) # c_clamped can be scalar or (n_embd,)
        constant_factor = 2 / (sf_x * c_clamped_sqrt + 1e-9) # Denom epsilon

        direction_factor = mob_addition_val / (addition_norm + 1e-9) # (B,T,n_embd)
        
        # Original: arg = torch.clamp((c * addition_norm) ** 0.5, ...)
        # (c * addition_norm^2)**0.5 = sqrt(c) * addition_norm (if c > 0)
        # c_clamped positive.
        arctanh_arg_base = c_clamped * addition_norm_sq # (B,T,1) or (B,T,n_embd)
        arctanh_arg = torch.sqrt(torch.abs(arctanh_arg_base) + 1e-9) # abs and epsilon
        
        arctanh_arg_clamped = torch.clamp(arctanh_arg, min=-0.9999, max=0.9999)
        return constant_factor * torch.arctanh(arctanh_arg_clamped) * direction_factor

def calculate_reference_point(x, per_head_curvature=False):
    """Calculate reference point for hyperbolic operations"""
    B, T, C = x.size()
    ref_point = torch.zeros_like(x[:, :1, :])
    if T > 1:
        ref_point = x[:, :-1, :]
        ref_point = F.pad(ref_point, (0, 0, 1, 0), mode='constant', value=0)
    return ref_point

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config # Store config for access to n_head later
        self.is_c_per_head = config.per_head_curvature # General intention, might be overridden by dynamic_curvature logic path

        if config.curvature_mode == 'fixed':
            self.c = config.curvature
            # Assuming fixed curvature from config is scalar. If it could be per-head, more logic needed.
            if isinstance(self.c, torch.Tensor) and self.c.numel() > 1:
                # This implies fixed curvature is per-head, check compatibility
                if self.c.shape[0] == config.n_head and self.is_c_per_head:
                    pass # Shape matches intention
                else:
                    # Mismatch or not intended to be per_head but has multiple values
                    print(f"Warning: Fixed curvature has {self.c.numel()} elements, but per_head_curvature is {self.is_c_per_head} or n_head mismatch.")
                    self.is_c_per_head = False # Treat as non-per-head or fallback
            else: # Scalar fixed curvature
                self.is_c_per_head = False
        elif config.curvature_mode == 'parametric':
            if config.curvature_initialization is None:
                curvature_init = 1.0
            else:
                block_idx = getattr(config, '_block_idx', 0)
                curvature_init = config.curvature_initialization[block_idx]
            self.c = nn.Parameter(torch.tensor(curvature_init).view(1))
            self.c.requires_grad = True
            self.is_c_per_head = False # Parametric c is scalar for the block
        elif config.curvature_mode == 'tied':
            # self.c will be overwritten by GPT.shared_curvature in _create_block
            # self.is_c_per_head is already set based on config.per_head_curvature.
            # The actual check for per-head operations will depend on the shape of self.c after it's set.
            self.c = torch.empty(0) # Placeholder, will be replaced by GPT module
        elif config.curvature_mode == 'random':
            if config.per_head_curvature:
                self.c = nn.Parameter(torch.rand(config.n_head)) # Shape: (n_head,)
                # self.is_c_per_head is already True from initial assignment
            else: # random, not per_head
                self.c = nn.Parameter(torch.rand(1)) # Shape: (1,)
                self.is_c_per_head = False # Explicitly scalar
        else:
            raise ValueError(f"Invalid curvature mode: {config.curvature_mode}")

        self.dynamic_curvature = config.dynamic_curvature
        if self.dynamic_curvature:
            if config.per_head_curvature: # This implicitly uses self.is_c_per_head logic
                self.curvature_predictor = nn.Linear(config.n_embd, config.n_head)
            else:
                self.curvature_predictor = nn.Linear(config.n_embd, 1)
            # If dynamic_curvature is true, the *final* curvature used in ops will be scalar (output of predictor).
            # So, per-head hyperbolic ops won't be used with the predictor's output.
            # self.is_c_per_head would refer to the base self.c if it were used, but it's overridden.

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # Input x is assumed to be in hyperbolic space (mapped in GPT.forward or previous block)
        
        n_head_for_ops = None
        current_c_for_ops = None

        if self.dynamic_curvature:
            # predicted_c will be (B, n_head) if self.is_c_per_head (i.e. config.per_head_curvature) is true,
            # else (B, 1).
            predicted_c = F.sigmoid(self.curvature_predictor(x[:, -1, :])) 
            current_c_for_ops = predicted_c # Shape (B, n_head) or (B, 1)
            self.last_dynamic_c = current_c_for_ops.detach() # Store for logging

            if self.is_c_per_head: # True if config.per_head_curvature was true
                # This implies predictor output is (B, n_head)
                assert predicted_c.ndim == 2 and predicted_c.shape[1] == self.config.n_head, \
                    f"Dynamic per-head curvature shape mismatch. Expected (B, {self.config.n_head}), got {predicted_c.shape}"
                n_head_for_ops = self.config.n_head
            else: # Predictor output is (B, 1)
                assert predicted_c.ndim == 2 and predicted_c.shape[1] == 1, \
                    f"Dynamic scalar curvature shape mismatch. Expected (B, 1), got {predicted_c.shape}"
                # current_c_for_ops is (B,1). Hyperbolic ops will handle this in their non-per-head path.
                n_head_for_ops = None
        else:
            current_c_for_ops = self.c
            if hasattr(self, 'last_dynamic_c'): # Clear if not dynamic this round / first pass
                delattr(self, 'last_dynamic_c')
            # Check if the actual self.c parameter is per-head AND per_head_curvature was intended/set
            if self.is_c_per_head and \
               isinstance(current_c_for_ops, torch.Tensor) and \
               current_c_for_ops.ndim == 1 and current_c_for_ops.shape[0] == self.config.n_head:
                n_head_for_ops = self.config.n_head
            else:
                 # Ensure n_head_for_ops is None if conditions for per-head are not met
                 # (e.g. self.c is scalar, or tied but not per_head, or shape mismatch)
                 n_head_for_ops = None

        reference_point = calculate_reference_point(x)

        # Map to tangent space at reference point before attention
        x_tan = logmap(reference_point, x, current_c_for_ops, n_head=n_head_for_ops)
        attn_update_tan = self.attn(self.ln_1(x_tan))
        # Map attention update back to hyperbolic space relative to reference point
        attn_update_hyp = expmap(reference_point, attn_update_tan, current_c_for_ops, n_head=n_head_for_ops)
        # Add residual using Mobius addition
        x = mobius_addition(x, attn_update_hyp, current_c_for_ops, n_head=n_head_for_ops)

        x_tan_mlp = logmap(reference_point, x, current_c_for_ops, n_head=n_head_for_ops)
        mlp_update_tan = self.mlp(self.ln_2(x_tan_mlp))
        # Map MLP update back to hyperbolic space relative to its reference point
        mlp_update_hyp = expmap(reference_point, mlp_update_tan, current_c_for_ops, n_head=n_head_for_ops)
        # Add residual using Mobius addition
        x = mobius_addition(x, mlp_update_hyp, current_c_for_ops, n_head=n_head_for_ops)

        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    curvature_mode: str = 'random' # 'fixed', 'parametric', 'tied', or any other value for random init
    curvature: float = 0.0 # Fixed curvature value when curvature_mode is 'fixed'
    curvature_initialization: list = field(default_factory=list) # List of initial curvature values for parametric mode (one per layer when provided)
    per_head_curvature: bool = True # whether to use a different curvature for each head
    use_embedding_curvature: bool = True #whether to use a curvature element also for the embedding layer. 
    dynamic_curvature: bool = True #whether to predict curvature based on input for the model. 
    def __post_init__(self):
        # Initialize curvature_initialization if it's empty
        if not self.curvature_initialization:
            self.curvature_initialization = [1.0-1.0e-3] + [1.0e-3] * (self.n_layer - 1)
    
    def set_layer_curvatures(self, curvature_values):
        """
        Set per-layer curvature initialization values for parametric mode.
        
        Args:
            curvature_values: Either a list of values (one per layer) or a single value to use for all layers
        """
        if self.curvature_mode != 'parametric':
            print("Warning: Setting layer curvatures but mode is not 'parametric'")
            
        if isinstance(curvature_values, (int, float)):
            # If a single value is provided, repeat it for all layers
            self.curvature_initialization = [float(curvature_values)] * self.n_layer
        else:
            # Expect a list with length equal to n_layer
        
            assert len(curvature_values) == self.n_layer, f"Expected {self.n_layer} curvature values, got {len(curvature_values)}"
            self.curvature_initialization = list(curvature_values)

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Create shared curvature parameter if using tied mode
        self.shared_curvature = None
        if config.curvature_mode == 'tied':
            if config.per_head_curvature:
                # Create one parameter per head, shared across all blocks
                self.shared_curvature = nn.Parameter(torch.rand(config.n_head)) # Shape: (n_head,)
            else:
                # Create a single parameter shared across all blocks
                self.shared_curvature = nn.Parameter(torch.tensor(1.0).view(1))
        if config.use_embedding_curvature: 
            self.embedding_curvature = nn.Parameter(torch.tensor(torch.rand(1)))
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([self._create_block(config, i) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _create_block(self, config, idx):
        # Create a new config object with the block index set
        import copy
        block_config = copy.copy(config)
        block_config._block_idx = idx
        
        # Create the block
        block = Block(block_config)
        
        # Pass the shared curvature parameter if using tied mode
        if config.curvature_mode == 'tied' and self.shared_curvature is not None:
            # For tied mode, we update the block's curvature parameter.
            # The block will then use this c, and its is_c_per_head flag (set from config)
            # will determine if per-head hyperbolic ops are used.
            block.c = self.shared_curvature
            
        return block

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.uniform_(module, a=0.0, b=1.0)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        if self.config.use_embedding_curvature:
            reference_point = calculate_reference_point(x)
            x = expmap(reference_point, x, self.embedding_curvature)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
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
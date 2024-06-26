from functools import partial
from itertools import islice, cycle
from inspect import isfunction
from math import ceil

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# helpers

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, 'n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

class SinuEmb(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        sinusoid_inp = torch.einsum('i , j -> i j', t, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim = -1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        n = x.shape[1]
        return self.emb[:n]

def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def cast_tuple(val, depth = 1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# attention

class Attention(nn.Module):
    def __init__(self, dim, seq_len, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def apply_rel_pos_emb(self, q, k, rel_pos_emb = None):
        if not exists(rel_pos_emb):
            return q, k

        (cls_q, q), (cls_k, k) = map(lambda t: (t[:, :, :1], t[:, :, 1:]), (q, k))
        q, k = apply_rotary_pos_emb(q, k, rel_pos_emb)
        q, k = map(lambda t: torch.cat(t, dim = -2), ((cls_q, q), (cls_k, k)))
        return q, k

    def forward(self, x, mask = None, rel_pos_emb = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if exists(rel_pos_emb):
            q, k = self.apply_rel_pos_emb(q, k, rel_pos_emb)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out, attn

# microsoft sparse attention CUDA kernel

# class SparseAttention(Attention):
#     def __init__(
#         self,
#         *args,
#         block_size = 16,
#         num_random_blocks = None,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         from deepspeed.ops.sparse_attention import SparseSelfAttention, VariableSparsityConfig
#         self.block_size = block_size

#         num_random_blocks = default(num_random_blocks, self.seq_len // block_size // 4)

#         self.attn_fn = SparseSelfAttention(
#             sparsity_config = VariableSparsityConfig(
#                 num_heads = self.heads,
#                 block = self.block_size,
#                 num_random_blocks = num_random_blocks,
#                 attention = 'unidirectional' if self.causal else 'bidirectional'
#             ),
#             max_seq_length = self.seq_len,
#             attn_mask_mode = 'add'
#         )

#     def forward(self, x, mask = None, rel_pos_emb = None):
#         b, n, _, h, device = *x.shape, self.heads, x.device
#         remainder = n % self.block_size
#         mask = default(mask, lambda: torch.ones(b, n, device = device).bool())

#         if remainder > 0:
#             padding = self.block_size - remainder
#             x = F.pad(x, (0, 0, 0, padding), value = 0)
#             mask = F.pad(mask, (0, padding), value = False)

#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

#         q, k = self.apply_rel_pos_emb(q, k, rel_pos_emb)

#         key_pad_mask = None
#         if exists(mask):
#             key_pad_mask = ~mask

#         out = self.attn_fn(q, k, v, key_padding_mask = key_pad_mask)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)
#         return out[:, :n]

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0., mult = 4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_types = None,
        image_fmap_size = None,
        sparse_attn = False,
        rel_pos_emb = True
    ):
        super().__init__()
        # self.token_emb = nn.Embedding(num_tokens, dim)
        self.cls_token = nn.Parameter(torch.randn(dim))

        # positional embeddings

        self.pos_emb = SinuEmb(dim, seq_len + 1)
        self.rel_pos_emb = SinuEmb(dim_head, seq_len) if rel_pos_emb else None

        # layers

        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)

        attn_types = default(attn_types, ('full',))
        attn_types = cast_tuple(attn_types)
        attn_type_layer = islice(cycle(attn_types), depth)

        for ind, sparse_attn, attn_type in zip(range(depth), sparse_layer, attn_type_layer):
            if attn_type == 'full':
                attn_class = Attention
            elif attn_type == 'sparse':
                attn_class = SparseAttention
            else:
                raise ValueError(f'attention type "{attn_type}" is not valid')

            layers.append(nn.ModuleList([
                PreNorm(dim, attn_class(dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.layers = layers
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, attention_mask = None):
        attention_mask = attention_mask.bool()
        b, n, _, = x.shape
        # x = self.token_emb(x)
        rel_pos_emb = self.rel_pos_emb(x) if exists(self.rel_pos_emb) else None

        cls_tokens = repeat(self.cls_token, 'd -> b () d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        if exists(attention_mask):
            attention_mask = F.pad(attention_mask, (1, 0), value = True)

        pos_emb = self.pos_emb(x)
        x += rearrange(pos_emb, 'n d -> () n d')

        for attn, ff in self.layers:
            hidden, attn_weights = attn(x, mask = attention_mask, rel_pos_emb = rel_pos_emb)
            x = hidden + x
            x = ff(x) + x

        return self.norm(x[:, 0]), attn_weights
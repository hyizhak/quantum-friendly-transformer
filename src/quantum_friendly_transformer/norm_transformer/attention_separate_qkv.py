"""Multi-head attention with separately parametrizable Q, K and V projections."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class AttentionWithSeparateQKV(nn.Module):
    """Use the ``nn.MultiheadAttention`` layout and mask conventions.

    Boolean masks mark blocked entries. Floating masks are additive. RoPE
    operates on projected [batch, sequence, heads, head_dim] tensors. Attention,
    padding and causal masks are combined. Nested tensors are not supported.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, rope=None, device=None, dtype=None):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1]")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = embed_dim if kdim is None else kdim
        self.vdim = embed_dim if vdim is None else vdim
        self.batch_first, self.dropout, self.rope = batch_first, dropout, rope
        self.add_zero_attn = add_zero_attn
        factory = {"device": device, "dtype": dtype}
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=bias, **factory)
        self.k_linear = nn.Linear(self.kdim, embed_dim, bias=bias, **factory)
        self.v_linear = nn.Linear(self.vdim, embed_dim, bias=bias, **factory)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory)
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim, **factory))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim, **factory))
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)
        self._reset_parameters()

    def _reset_parameters(self):
        for projection in (self.q_linear, self.k_linear, self.v_linear):
            nn.init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)

    @staticmethod
    def _additive_mask(mask, reference):
        if mask.dtype == torch.bool:
            return torch.zeros(mask.shape, device=reference.device, dtype=reference.dtype).masked_fill(
                mask.to(reference.device), float("-inf"))
        if not torch.is_floating_point(mask):
            raise ValueError("Attention masks must be boolean or floating point")
        return mask.to(reference)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        unbatched = query.ndim == 2
        if query.ndim not in (2, 3) or key.ndim != query.ndim or value.ndim != query.ndim:
            raise ValueError("Query, key and value must all be 2D or all be 3D")
        if unbatched:
            query, key, value = (x.unsqueeze(0) for x in (query, key, value))
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        elif not self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        output, weights = self.multi_head_attention_forward(
            query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights, is_causal=is_causal)
        if unbatched:
            return output.squeeze(0), None if weights is None else weights.squeeze(0)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output, weights

    def multi_head_attention_forward(self, query, key, value, key_padding_mask=None,
                                     need_weights=True, attn_mask=None,
                                     average_attn_weights=True, is_causal=False):
        """Internal batch-first implementation (also used by legacy callers)."""
        batch, target, _ = query.shape
        source = key.shape[1]
        if key.shape[:2] != value.shape[:2] or key.shape[0] != batch:
            raise ValueError("Key/value sequence lengths and batch sizes must agree")
        q, k, v = (projection(x).reshape(batch, x.shape[1], self.num_heads, self.head_dim)
                   for projection, x in ((self.q_linear, query), (self.k_linear, key),
                                         (self.v_linear, value)))
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        mask = None
        if attn_mask is not None:
            mask = self._additive_mask(attn_mask, q)
            if mask.shape == (target, source):
                mask = mask[None, None]
            elif mask.shape == (batch * self.num_heads, target, source):
                mask = mask.reshape(batch, self.num_heads, target, source)
            else:
                raise ValueError("attn_mask must have shape [L, S] or [batch*heads, L, S]")
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, source):
                raise ValueError("key_padding_mask must have shape [batch, source_length]")
            padding = self._additive_mask(key_padding_mask, q)[:, None, None, :]
            mask = padding if mask is None else mask + padding
        if is_causal:
            causal = torch.ones(target, source, device=q.device, dtype=torch.bool).triu(1)
            causal = self._additive_mask(causal, q)[None, None]
            mask = causal if mask is None else mask + causal
        if self.bias_k is not None:
            bias_k = self.bias_k.reshape(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            bias_v = self.bias_v.reshape(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = torch.cat((k, bias_k.expand(batch, -1, -1, -1)), dim=2)
            v = torch.cat((v, bias_v.expand(batch, -1, -1, -1)), dim=2)
            if mask is not None:
                mask = F.pad(mask, (0, 1))
        if self.add_zero_attn:
            zero = k.new_zeros(batch, self.num_heads, 1, self.head_dim)
            k, v = torch.cat((k, zero), dim=2), torch.cat((v, zero), dim=2)
            if mask is not None:
                mask = F.pad(mask, (0, 1))
        dropout = self.dropout if self.training else 0.0
        weights = None
        if need_weights:
            scores = (q / math.sqrt(self.head_dim)) @ k.transpose(-2, -1)
            if mask is not None:
                scores = scores + mask
            all_blocked = scores.isneginf().all(dim=-1, keepdim=True)
            scores = scores.masked_fill(all_blocked, 0)
            probabilities = scores.softmax(dim=-1).masked_fill(all_blocked, 0)
            probabilities = F.dropout(probabilities, p=dropout, training=self.training)
            attended = probabilities @ v
            weights = probabilities.mean(dim=1) if average_attn_weights else probabilities
        else:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)
        attended = attended.transpose(1, 2).reshape(batch, target, self.embed_dim)
        return self.out_proj(attended), weights

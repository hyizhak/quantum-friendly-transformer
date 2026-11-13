import math
import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        """
        Args:
            head_dim (int): Dimension of each attention head (must be even).
            max_seq_len (int): Maximum sequence length for which to precompute the embeddings.
            base (float): Base factor for computing inverse frequencies.
        """
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be positive and even")
        if max_seq_len <= 0 or not math.isfinite(base) or base <= 0:
            raise ValueError("max_seq_len and base must be positive and finite")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Precompute and register the rotary embeddings as buffers (they don't update during training)
        cos, sin = self._get_rotary_embeddings(max_seq_len, head_dim, base)
        self.register_buffer("cos", cos)  # shape: [max_seq_len, head_dim]
        self.register_buffer("sin", sin)  # shape: [max_seq_len, head_dim]

    def _get_rotary_embeddings(self, seq_len, head_dim, base):
        # Compute inverse frequencies for even indices
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # positions: shape [seq_len]
        positions = torch.arange(seq_len, dtype=torch.float32)
        # Outer product: shape [seq_len, head_dim/2]
        sinusoid_inp = torch.outer(positions, inv_freq)
        sin = torch.sin(sinusoid_inp)  # shape: [seq_len, head_dim/2]
        cos = torch.cos(sinusoid_inp)  # shape: [seq_len, head_dim/2]
        # rotate_half pairs i with i + head_dim/2; both need the same angle.
        sin = torch.cat((sin, sin), dim=-1)
        cos = torch.cat((cos, cos), dim=-1)
        return cos, sin

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Rebuild deterministic caches: old checkpoints used incorrect pairing.
        cos, sin = self._get_rotary_embeddings(self.max_seq_len, self.head_dim, self.base)
        state_dict[prefix + "cos"] = cos.to(self.cos)
        state_dict[prefix + "sin"] = sin.to(self.sin)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def rotate_half(self, x):
        """
        Splits the last dimension into two halves and then concatenates them
        after flipping the sign on the second half.
        
        Args:
            x (Tensor): Last dimension of size head_dim.
            
        Returns:
            Tensor: Rotated tensor with the same shape as x.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, t, seq_len=None):
        """
        Applies the rotary positional embeddings to tensor t.
        
        Args:
            t (Tensor): Input tensor, expected shape [batch, seq_len, n_heads, head_dim].
            seq_len (int, optional): Sequence length to use from precomputed cos/sin.
                If None, uses t.shape[1].
                
        Returns:
            Tensor: Tensor with RoPE applied, same shape as t.
        """
        if t.ndim != 4 or t.shape[-1] != self.head_dim:
            raise ValueError("Expected [batch, sequence, heads, head_dim]")
        if seq_len is None:
            seq_len = t.shape[1]
        if seq_len != t.shape[1] or not 0 < seq_len <= self.max_seq_len:
            raise ValueError("Sequence length must match the input and fit the rotary cache")
        # Slice precomputed embeddings for the current sequence length
        cos = self.cos[:seq_len, :].to(t)  # shape: [seq_len, head_dim]
        sin = self.sin[:seq_len, :].to(t)  # shape: [seq_len, head_dim]
        # Reshape for broadcasting: [1, seq_len, 1, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        return t * cos + self.rotate_half(t) * sin

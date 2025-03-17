import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.nn.utils.parametrizations import spectral_norm
from .attention_separate_qkv import AttentionWithSeparateQKV

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        """
        Args:
            head_dim (int): Dimension of each attention head (must be even).
            max_seq_len (int): Maximum sequence length for which to precompute the embeddings.
            base (float): Base factor for computing inverse frequencies.
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
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
        # Interleave to create shape: [seq_len, head_dim]
        sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, head_dim)
        cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, head_dim)
        return cos, sin

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
        if seq_len is None:
            seq_len = t.shape[1]
        # Slice precomputed embeddings for the current sequence length
        cos = self.cos[:seq_len, :]  # shape: [seq_len, head_dim]
        sin = self.sin[:seq_len, :]  # shape: [seq_len, head_dim]
        # Reshape for broadcasting: [1, seq_len, 1, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        return t * cos + self.rotate_half(t) * sin

class SpectrallyNormalizedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_emb, max_seq_len, apply_embedding_sn=False,
                 apply_attention_sn=False, apply_ffn_sn=False, embedding_layer=None):
        super().__init__()
        
        # Embedding layer
        if embedding_layer is not None:
            self.embedding = embedding_layer
            assert self.embedding.weight.shape[1] == d_model, "Embedding dimension must match d_model"
        else:
            self.embedding = nn.Embedding(num_embeddings=num_emb, embedding_dim=d_model, padding_idx=0)

        if apply_embedding_sn:
            # consider wrapping the weight manually.
            self.embedding.weight.data = self.embedding.weight.data / torch.norm(self.embedding.weight.data, dim=1, keepdim=True)

        self.RoPE = RotaryPositionalEmbedding(head_dim=d_model, max_seq_len=max_seq_len)
        
        # Self-attention components
        self.attn = AttentionWithSeparateQKV(embed_dim=d_model, num_heads=nhead)
        if apply_attention_sn:
            # Apply SN to the projection matrices for Q, K, V manually if needed.
            self.attn.q_linear = spectral_norm(self.attn.q_linear)
            self.attn.k_linear = spectral_norm(self.attn.k_linear)
            self.attn.v_linear = spectral_norm(self.attn.v_linear)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        if apply_ffn_sn:
            for layer in self.ffn:
                if isinstance(layer, nn.Linear):
                    # Apply spectral normalization
                    layer = spectral_norm(layer)
        
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # If using an embedding layer, x will be token indices.
        if x.dtype == torch.long:
            x = self.embedding(x)

        x = x.unsqueeze(2)  # Add an n_head dimension
        x = self.RoPE(x)
        
        # Self-attention expects (seq_len, batch_size, d_model)
        x = x.squeeze(2).transpose(0, 1)

        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + attn_output  # Residual connection
        
        # Feed-forward network
        x = x.transpose(0, 1)
        x = x + self.ffn(x)
        return x
    
    def get_input_embeddings(self):
        return self.embedding
    
class SpectrallyNormalizedTransformerForSequenceClassification(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_emb, max_seq_len, num_classes, apply_embedding_sn=False,
                 apply_attention_sn=False, apply_ffn_sn=False, embedding_layer=None):
        super().__init__()
        
        self.transformer = SpectrallyNormalizedTransformerBlock(
            d_model=d_model, nhead=nhead, d_ff=d_ff, num_emb=num_emb, max_seq_len=max_seq_len,
            apply_embedding_sn=apply_embedding_sn,
            apply_attention_sn=apply_attention_sn,
            apply_ffn_sn=apply_ffn_sn,
            embedding_layer=embedding_layer
        )
        
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        x = self.transformer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        cls_repr = x[:, 0, :]
        return self.classifier(cls_repr)
    
class SpectrallyNormalizedTransformerForTokenClassification(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_emb, max_seq_len, num_classes, apply_embedding_sn=False,
                 apply_attention_sn=False, apply_ffn_sn=False, embedding_layer=None):
        super().__init__()
        
        self.transformer = SpectrallyNormalizedTransformerBlock(
            d_model=d_model, nhead=nhead, d_ff=d_ff, num_emb=num_emb, max_seq_len=max_seq_len,
            apply_embedding_sn=apply_embedding_sn,
            apply_attention_sn=apply_attention_sn,
            apply_ffn_sn=apply_ffn_sn,
            embedding_layer=embedding_layer
        )
        
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        x = self.transformer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return self.classifier(x)

# Example instantiation:
if __name__ == "__main__":
    model_variant = SpectrallyNormalizedTransformerBlock(
        d_model=512, nhead=8, d_ff=2048, num_emb=7, max_seq_len=256,
        apply_embedding_sn=True,
        apply_attention_sn=True,
        apply_ffn_sn=True
    )
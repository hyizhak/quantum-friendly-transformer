import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.nn.utils.parametrizations import spectral_norm
from .attention_separate_qkv import AttentionWithSeparateQKV
from .rope import RotaryPositionalEmbedding

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

        self.RoPE = RotaryPositionalEmbedding(head_dim=d_model//nhead, max_seq_len=max_seq_len)
        
        # Self-attention components
        self.attn = AttentionWithSeparateQKV(embed_dim=d_model, num_heads=nhead, rope=self.RoPE)
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
        
        # Self-attention expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)

        attn_output, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x.transpose(0, 1)
        x = x + attn_output  # Residual connection
        
        # Feed-forward network
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
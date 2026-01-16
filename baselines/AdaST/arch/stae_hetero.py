import torch
import torch.nn as nn
from typing import Optional

import pdb


class SpatialMixer(nn.Module):
    """
    Simple learned spatial mixing over nodes.

    Input:  x  (B*L, N, d)
    Output: y  (B*L, N, d)
    """
    def __init__(self, N: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.N = N
        # Initialize near-zero so softmax(A) starts ~uniform; scale by 1/N
        self.A = nn.Parameter(torch.randn(N, N) * (1.0 / N))
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*L, N, d)
        A = torch.softmax(self.A, dim=-1)         # (N, N)
        y = torch.einsum('nm,bmd->bnd', A, x)     # (B*L, N, d)
        y = 0.5 * (y + x)                         # mild residual blend
        y = self.drop(self.proj(y))
        return y


class TemporalConvMixer(nn.Module):
    """
    Depthwise-separable 1D conv along time; no attention.

    Input:  x  (B*N, L, d)
    Output: y  (B*N, L, d)
    """
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 1 == 0
        self.causal = causal
        padding = (kernel_size - 1) if causal else (kernel_size // 2)

        self.dw = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=padding)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, L, d)
        x_t = x.transpose(1, 2)        # (B*N, d, L)
        y = self.dw(x_t)               # (B*N, d, L + pad)
        if self.causal:
            y = y[..., : x_t.size(-1)] # trim any future leakage
        y = self.pw(self.act(y))
        y = self.drop(y).transpose(1, 2)
        return x + y                   # residual


class TemporalMixerLayer(nn.Module):
    """
    Temporal mixing (depthwise temporal conv) + FFN with pre-norm-like residuals.

    Input/Output: (B, L, N, d)
    """
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 2048,
        kernel_size: int = 3,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.mixer = TemporalConvMixer(model_dim, kernel_size, dropout, causal)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N, d)
        B, L, N, D = x.shape

        # (B*N, L, d) for temporal conv
        x_reshaped = x.transpose(1, 2).reshape(B * N, L, D)

        # Temporal mixer + residual + LN
        residual = x_reshaped
        out = self.mixer(x_reshaped)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        # FFN + residual + LN
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        # Back to (B, L, N, d)
        out = out.reshape(B, N, L, D).transpose(1, 2)
        return out


class SpatialMixerLayer(nn.Module):
    """
    Spatial mixing over nodes + FFN with residuals.

    Input/Output: (B, L, N, d)
    """
    def __init__(
        self,
        num_nodes: int,
        model_dim: int,
        feed_forward_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mixer = SpatialMixer(num_nodes, model_dim, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N, d)
        B, L, N, D = x.shape

        # (B*L, N, d) for spatial mixing
        x_reshaped = x.reshape(B * L, N, D)

        residual = x_reshaped
        out = self.mixer(x_reshaped)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.reshape(B, L, N, D)
        return out


class AttentionLayer(nn.Module):
    """
    Multi-head dot-product attention across a specified length dimension.
    Expects tensors permuted so the length dimension is second-to-last: (..., length, d)

    Q: (B, ..., T, d)
    K,V: (B, ..., S, d)
    """
    def __init__(self, model_dim: int, num_heads: int = 8, mask: bool = False):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B = query.shape[0]
        T = query.shape[-2]
        S = key.shape[-2]

        q = self.FC_Q(query)
        k = self.FC_K(key)
        v = self.FC_V(value)

        # Merge heads by concatenating along batch dimension for efficiency
        q = torch.cat(torch.split(q, self.head_dim, dim=-1), dim=0)   # (H*B, ..., T, d_h)
        k = torch.cat(torch.split(k, self.head_dim, dim=-1), dim=0)   # (H*B, ..., S, d_h)
        v = torch.cat(torch.split(v, self.head_dim, dim=-1), dim=0)

        kT = k.transpose(-1, -2)                                      # (H*B, ..., d_h, S)
        attn = (q @ kT) / (self.head_dim ** 0.5)                      # (H*B, ..., T, S)

        if self.mask:
            mask = torch.ones(T, S, dtype=torch.bool, device=attn.device).tril()
            attn = attn.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v                                                # (H*B, ..., T, d_h)

        # Restore heads by concatenating back on feature dim
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)           # (B, ..., T, d)
        out = self.out_proj(out)
        return out


class SelfAttentionLayer(nn.Module):
    """
    Pre-norm self-attention + FFN with residuals.

    x: (B, ..., L, d) after transpose; restored on exit.
    """
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        mask: bool = False,
    ):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        # Move target length dim to second-to-last so attn runs along it
        x = x.transpose(dim, -2)                          # (..., L, d)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAE(nn.Module):
    """
    STAEformer-style model with Spatial & Temporal mixers and optional self-attention.
    (Ref: STAEformer CIKM 2023; this variant keeps the original meaning but fixes wiring)

    history_data: (B, in_steps, N, input_dim + [tod,dow as fractional features])
      - channel 0..input_dim-1: primary inputs
      - if tod_embedding_dim>0, channel input_dim is fractional TOD in [0,1) scaled by steps_per_day
      - if dow_embedding_dim>0, channel input_dim+1 is fractional DOW in [0,1) scaled by 7
    """
    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 0,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_heads: int = 4,
        use_mixed_proj: bool = True,
        temporal_kernel_size: int = 3,
        temporal_causal: bool = False,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        # Feature channel model width (per token)
        self.concat_dim = (
            input_embedding_dim 
            + (tod_embedding_dim if tod_embedding_dim > 0 else 0)
            + (dow_embedding_dim if dow_embedding_dim > 0 else 0)
            + (spatial_embedding_dim if spatial_embedding_dim > 0 else 0)
            + (adaptive_embedding_dim if adaptive_embedding_dim > 0 else 0)
        )
        self.model_dim = 32

        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        # Input/token projections
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.model_proj = nn.Linear(self.concat_dim, self.model_dim)

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        else:
            self.tod_embedding = None

        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        else:
            self.dow_embedding = None

        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        else:
            self.node_emb = None

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)
        else:
            self.adaptive_embedding = None

        # Three modulation/projection heads (produce 3 * model_dim from [x || emb])
        concat_in_tod = input_embedding_dim + (tod_embedding_dim if tod_embedding_dim > 0 else 0)
        concat_in_spa = input_embedding_dim + (spatial_embedding_dim if spatial_embedding_dim > 0 else 0)
        concat_in_adp = input_embedding_dim + (adaptive_embedding_dim if adaptive_embedding_dim > 0 else 0)

        self.tod_proj = nn.Linear(concat_in_tod, 3 * self.model_dim) if concat_in_tod > 0 else None
        self.spatial_proj = nn.Linear(concat_in_spa, 3 * self.model_dim) if concat_in_spa > 0 else None
        self.adaptive_proj = nn.Linear(concat_in_adp, 3 * self.model_dim) if concat_in_adp > 0 else None

        # Spatial stacks
        self.mixer_layers_s = nn.ModuleList([
            SpatialMixerLayer(num_nodes, self.model_dim, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.mixer_layers_s_specific = nn.ModuleList([
            SpatialMixerLayer(num_nodes, self.model_dim, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])

        # Temporal stacks (self-attention across time)
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.attn_layers_t_specific = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Gating across {shared, temporal-specific, spatial-specific}
        self.gate_proj = nn.Linear(self.model_dim * 3, 3)

        # Output head(s)
        if use_mixed_proj:
            self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, output_dim)

    def _maybe_cat(self, x_a: Optional[torch.Tensor], x_b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_a is None and x_b is None:
            return None
        if x_a is None:
            return x_b
        if x_b is None:
            return x_a
        return torch.cat([x_a, x_b], dim=-1)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: Optional[torch.Tensor] = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        history_data: (B, in_steps, N, input_dim + [tod?, dow?])
        Returns:
            (B, out_steps, N, output_dim)
        """
        x_raw = history_data
        B, L, N, C = x_raw.shape
        assert N == self.num_nodes, "num_nodes mismatch"

        # Parse channels
        x_in = x_raw[..., : self.input_dim]                                 # (B, L, N, input_dim)
        idx = 1

        tod_idx = None
        dow_idx = None
        if self.tod_embedding is not None:
            tod_idx = idx
            idx += 1
        if self.dow_embedding is not None:
            dow_idx = idx
            idx += 1

        # Base token embedding
        x = self.input_proj(x_in)                                           # (B, L, N, d_in)

        # Optional embeddings
        if self.tod_embedding is not None:
            # Expect fractional TOD in [0,1); scale to [0, steps_per_day-1]
            tod_frac = x_raw[..., tod_idx].clamp(min=0.0, max=0.999999)
            tod = (tod_frac * self.steps_per_day).long()
            tod_emb = self.tod_embedding(tod)                               # (B, L, N, d_tod)
        else:
            tod_emb = None

        if self.dow_embedding is not None:
            # Expect fractional DOW in [0,1); scale to [0,6]
            dow_frac = x_raw[..., dow_idx].clamp(min=0.0, max=0.999999)
            dow = (dow_frac * 7).long()
            dow_emb = self.dow_embedding(dow)                               # (B, L, N, d_dow)
        else:
            dow_emb = None

        if self.node_emb is not None:
            spatial_emb = self.node_emb.unsqueeze(0).unsqueeze(0).expand(B, L, N, -1)
        else:
            spatial_emb = None

        if self.adaptive_embedding is not None:
            adp_emb = self.adaptive_embedding.unsqueeze(0).expand(B, -1, -1, -1)  # (B, L, N, d_adp)
        else:
            adp_emb = None

        # Concatenate available additive embeddings into x_base
        x_base = x
        if tod_emb is not None:
            x_base = self._maybe_cat(x_base, tod_emb)
        if dow_emb is not None:
            x_base = self._maybe_cat(x_base, dow_emb)
        if spatial_emb is not None:
            x_base = self._maybe_cat(x_base, spatial_emb)
        if adp_emb is not None:
            x_base = self._maybe_cat(x_base, adp_emb)

        # project back to model_dim
        x_base = self.model_proj(x_base)                                   # (B, L, N, D)

        # Build three modulated streams from available heads
        def triple_from_proj(proj: Optional[nn.Linear], emb: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if proj is None:
                return None
            cat = self._maybe_cat(x, emb)
            if cat is None:
                return None
            cat = proj(cat)                                           # (B, L, N, 3*D)
            # sigmoid
            cat = torch.sigmoid(cat)
            return cat

        x_all_tod = triple_from_proj(self.tod_proj, tod_emb)
        x_all_spa = triple_from_proj(self.spatial_proj, spatial_emb)
        x_all_adp = triple_from_proj(self.adaptive_proj, adp_emb)

        # Split into (base, temporal, spatial) per head; sum available heads
        zeros = torch.zeros(B, L, N, self.model_dim, device=x_base.device, dtype=x_base.dtype)

        def split3(t: Optional[torch.Tensor]):
            if t is None: return (zeros, zeros, zeros)
            return torch.split(t, self.model_dim, dim=-1)

        x_tod, x_t_tod, x_s_tod = split3(x_all_tod)
        x_spa, x_t_spa, x_s_spa = split3(x_all_spa)
        x_adp, x_t_adp, x_s_adp = split3(x_all_adp)

        x_shared = x_base * (x_tod + x_spa + x_adp + 1e-6)                  # (B,L,N,D)
        x_time   = x_base * (x_t_tod + x_t_spa + x_t_adp + 1e-6)
        x_space  = x_base * (x_s_tod + x_s_spa + x_s_adp + 1e-6)

        # Shared temporal+spatial stacks
        x_shared_out = x_shared
        for attn in self.attn_layers_t:
            x_shared_out = attn(x_shared_out, dim=1)                         # along time
        for mix_s in self.mixer_layers_s:
            x_shared_out = mix_s(x_shared_out)                               # along nodes

        # Temporal-specific stack
        x_t_out = x_time
        for attn in self.attn_layers_t_specific:
            x_t_out = attn(x_t_out, dim=1)

        # Spatial-specific stack
        x_s_out = x_space
        for mix_s in self.mixer_layers_s_specific:
            x_s_out = mix_s(x_s_out)

        # Gated fusion
        # combined = torch.cat([x_shared_out, x_t_out, x_s_out], dim=-1)       # (B, L, N, 3D)
        # gates = torch.softmax(self.gate_proj(combined), dim=-1)              # (B, L, N, 3)
        # x_gated = (
        #     gates[..., 0:1] * x_shared_out
        #     + gates[..., 1:2] * x_t_out
        #     + gates[..., 2:3] * x_s_out
        # )                                                                     # (B, L, N, D)
        x_gated = (x_shared_out + x_t_out + x_s_out) / 3.0                     # (B, L, N, D)

        # Output projection
        if self.use_mixed_proj:
            out = x_gated.transpose(1, 2).reshape(B, N, L * self.model_dim)  # (B, N, L*D)
            out = self.output_proj(out).view(B, N, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)                                         # (B, out_steps, N, output_dim)
        else:
            out = x_gated.transpose(1, 3)                                     # (B, D, N, L)
            out = self.temporal_proj(out)                                     # (B, D, N, out_steps)
            out = self.output_proj(out.transpose(1, 3))                       # (B, out_steps, N, output_dim)

        return out

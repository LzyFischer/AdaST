import torch
import torch.nn as nn
import torch.nn.functional as F


class CombineCorrelationAggregation(nn.Module):
    def __init__(self, model_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(model_dim // reduction, 8)
        self.gate_net = nn.Sequential(
            nn.Linear(model_dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3)
        )
    
    def forward(self, x, x_t, x_s):
        B, T, S, D = x.shape
        x_t_reshaped = x_t.permute(0, 2, 1, 3).reshape(-1, x_t.shape[1], x_t.shape[3])
    
        x_t_norm = F.normalize(x_t_reshaped, p=2, dim=-1)
        similarity_t = torch.sum(x_t_norm[:, :-1, :] * x_t_norm[:, 1:, :], dim=-1)
        corr_x_t = similarity_t.reshape(B, -1).mean(dim=-1)
        
        x_s_reshaped = x_s.permute(0, 1, 2, 3).reshape(-1, x_s.shape[2], x_s.shape[3])
        x_s_norm = F.normalize(x_s_reshaped, p=2, dim=-1)
        similarity_s = torch.sum(x_s_norm[:, :-1, :] * x_s_norm[:, 1:, :], dim=-1)
        corr_x_s = similarity_s.reshape(B, -1).mean(dim=-1)

        x_reshaped = x.permute(0, 2, 1, 3).reshape(-1, x.shape[2], x.shape[3])
        x_norm = F.normalize(x_reshaped, p=2, dim=-1)
        similarity_x_t = torch.sum(x_norm[:, :-1, :] * x_norm[:, 1:, :], dim=-1)
        x_reshaped = x.permute(0, 1, 2, 3).reshape(-1, x.shape[2], x.shape[3])
        x_norm = F.normalize(x_reshaped, p=2, dim=-1)
        similarity_x_s = torch.sum(x_norm[:, :-1, :] * x_norm[:, 1:, :], dim=-1)
        corr_x = (similarity_x_t.reshape(B, -1).mean(dim=-1) + similarity_x_s.reshape(B, -1).mean(dim=-1)) / 2.0

        corr_features = torch.stack([corr_x, corr_x_t, corr_x_s], dim=-1)
        combined = torch.cat([x, x_t, x_s], dim=-1)
        gates = self.gate_net(combined)
        gates = gates * corr_features.view(B, 1, 1, -1)
        gates = torch.softmax(gates, dim=-1).unsqueeze(-2)  # (B, L, N, 1, 3)
        stacked = torch.stack([x, x_t, x_s], dim=-1)  # (B, L, N, D, 3)
        out = (stacked * gates).sum(dim=-1)
        
        # Return gates as well for analysis
        return out, gates.squeeze(-2), corr_features

def aggregation_in_model(aggregation_type='simple_avg', model_dim=128):
    if aggregation_type == 'combine_correlation':
        return CombineCorrelationAggregation(model_dim=model_dim)
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")


class SpatialMixer(nn.Module):
    def __init__(self, N: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.N = N
        self.A = nn.Parameter(torch.randn(N, N) * (1.0 / max(1, N)))
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = torch.softmax(self.A, dim=-1)
        y = torch.einsum('nm,bmd->bnd', A, x)
        y = 0.5 * (y + x)
        y = self.drop(self.proj(y))
        return y


class SpatialMixerLayer(nn.Module):
    def __init__(self, num_nodes, model_dim, feed_forward_dim=2048, dropout=0.1):
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

    def forward(self, x):
        B, L, N, D = x.shape
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
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        B = query.shape[0]
        Tq = query.shape[-2]
        Tk = key.shape[-2]

        Q = self.FC_Q(query)
        K = self.FC_K(key)
        V = self.FC_V(value)

        Q = torch.cat(torch.split(Q, self.head_dim, dim=-1), dim=0)
        K = torch.cat(torch.split(K, self.head_dim, dim=-1), dim=0)
        V = torch.cat(torch.split(V, self.head_dim, dim=-1), dim=0)

        K = K.transpose(-1, -2)
        attn = (Q @ K) / (self.head_dim ** 0.5)

        if self.mask:
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=attn.device).tril()
            attn.masked_fill_(~mask, -torch.inf)

        attn = torch.softmax(attn, dim=-1)
        out = attn @ V
        out = torch.cat(torch.split(out, B, dim=0), dim=-1)
        out = self.out_proj(out)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0.0, mask=False):
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

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
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


class AdaST(nn.Module):
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
        aggregation_mode: str = 'combine_correlation',
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

        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            adp = nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            nn.init.xavier_uniform_(adp)
            self.adaptive_embedding = adp

        dim_x = input_embedding_dim
        dim_tod = dim_x + (tod_embedding_dim if tod_embedding_dim > 0 else 0)
        dim_dow = dim_x + (dow_embedding_dim if dow_embedding_dim > 0 else 0)
        dim_spatial = dim_x + (spatial_embedding_dim if spatial_embedding_dim > 0 else 0)
        dim_adp = dim_x + (adaptive_embedding_dim if adaptive_embedding_dim > 0 else 0)

        self.model_dim = dim_tod + dim_dow + dim_spatial + dim_adp

        def three_way(in_dim: int) -> nn.Linear:
            return nn.Linear(in_dim, in_dim * 3)

        self.tod_proj = three_way(dim_tod)
        self.dow_proj = three_way(dim_dow)
        self.spatial_proj = three_way(dim_spatial)
        self.adp_proj = three_way(dim_adp)

        self.mixer_layers_s = nn.ModuleList(
            [SpatialMixerLayer(num_nodes, self.model_dim, feed_forward_dim, dropout) for _ in range(num_layers)]
        )
        self.mixer_layers_s_specific = nn.ModuleList(
            [SpatialMixerLayer(num_nodes, self.model_dim, feed_forward_dim, dropout) for _ in range(num_layers)]
        )
        self.attn_layers_t = nn.ModuleList(
            [SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.attn_layers_t_specific = nn.ModuleList(
            [SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.aggregation = aggregation_in_model(
            aggregation_type=aggregation_mode,
            model_dim=self.model_dim
        )

        if use_mixed_proj:
            self.output_proj = nn.Linear(in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, 
                return_embeddings: bool = False, **kwargs):
        """
        Args:
            history_data: (B, in_steps, N, input_dim + [tod] + [dow])
            return_embeddings: If True, returns embeddings and gate weights
        
        Returns:
            If return_embeddings is False: predictions (B, out_steps, N, output_dim)
            If return_embeddings is True: dict with 'prediction', 'embeddings', 'gate_weights'
        """
        x_full = history_data
        B = x_full.shape[0]

        if self.tod_embedding_dim > 0:
            tod_idx = (x_full[..., 1] * self.steps_per_day).clamp(min=0, max=self.steps_per_day - 1).long()
        if self.dow_embedding_dim > 0:
            dow_idx = (x_full[..., 2] * 7).clamp(min=0, max=6).long()

        x = x_full[..., : self.input_dim]
        x = self.input_proj(x)

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(tod_idx)
        else:
            tod_emb = None

        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow_idx)
        else:
            dow_emb = None

        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(B, self.in_steps, *self.node_emb.shape)
        else:
            spatial_emb = None

        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(B, *self.adaptive_embedding.shape)
        else:
            adp_emb = None

        x_tod = torch.cat([x, tod_emb], dim=-1) if tod_emb is not None else x
        x_dow = torch.cat([x, dow_emb], dim=-1) if dow_emb is not None else x
        x_spatial = torch.cat([x, spatial_emb], dim=-1) if spatial_emb is not None else x
        x_adp = torch.cat([x, adp_emb], dim=-1) if adp_emb is not None else x

        def three_split(proj, t):
            if proj is None:
                return t, t, t
            out = proj(t)
            a, b, c = out.chunk(3, dim=-1)
            return a, b, c

        x_tod_m, x_tod_t, x_tod_s = three_split(self.tod_proj, x_tod)
        x_dow_m, x_dow_t, x_dow_s = three_split(self.dow_proj, x_dow)
        x_spa_m, x_spa_t, x_spa_s = three_split(self.spatial_proj, x_spatial)
        x_adp_m, x_adp_t, x_adp_s = three_split(self.adp_proj, x_adp)

        x_main = torch.cat([x_tod_m, x_dow_m, x_spa_m, x_adp_m], dim=-1)
        x_t = torch.cat([x_tod_t, x_dow_t, x_spa_t, x_adp_t], dim=-1)
        x_s = torch.cat([x_tod_s, x_dow_s, x_spa_s, x_adp_s], dim=-1)

        x = x_main
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for mixer_s in self.mixer_layers_s:
            x = mixer_s(x)

        xt = x_t
        for attn in self.attn_layers_t_specific:
            xt = attn(xt, dim=1)

        xs = x_s
        for mixer_s in self.mixer_layers_s_specific:
            xs = mixer_s(xs)

        # Get aggregated output and gate weights
        x_agg, gate_weights, corr_features = self.aggregation(x, xt, xs)

        # Decode
        if self.use_mixed_proj:
            out = x_agg.transpose(1, 2)
            out = out.reshape(B, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(B, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)
        else:
            out = x_agg.transpose(1, 3)
            out = self.temporal_proj(out)
            out = self.output_proj(out.transpose(1, 3))

        if return_embeddings:
            return {
                'prediction': out,
                'embeddings': {
                    'st': x.detach(),      # Spatio-temporal branch (B, L, N, D)
                    's': xs.detach(),      # Spatial branch (B, L, N, D)
                    't': xt.detach()       # Temporal branch (B, L, N, D)
                },
                'gate_weights': gate_weights.detach(),  # (B, L, N, 3)
                'correlation_features': corr_features.detach()  # (B, 3)
            }
        
        return out
from typing import Optional, Tuple, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class BatchNormLastDim(nn.Module):
    """BatchNorm along the last feature dim.

    Supports both 3-D (B, S, d) and 4-D (B, N, L, d) tensors.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            _, S, _ = x.shape
            y = x.transpose(1, 2)  # (B, d, S)
            y = self.bn(y)
            return y.transpose(1, 2)
        elif x.dim() == 4:
            B, N, L, d = x.shape
            y = x.permute(0, 3, 1, 2).reshape(B, d, N * L)
            y = self.bn(y)
            y = y.reshape(B, d, N, L).permute(0, 2, 3, 1)
            return y
        else:
            raise ValueError("BatchNormLastDim expects 3D or 4D input.")


class NormChooser(nn.Module):
    """Selects LayerNorm or BatchNorm for a given feature dim."""

    def __init__(self, d_model: int, norm_type: str = 'layer') -> None:
        super().__init__()
        if norm_type == 'layer':
            self.norm: nn.Module = nn.LayerNorm(d_model)
        elif norm_type == 'batch':
            self.norm = BatchNormLastDim(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class SpatialTemporalGating(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.strength_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def mlp_gating(
        self, emb: torch.Tensor, infor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        S = self.strength_mlp(infor)
        out_emb = (emb - infor) * S
        return out_emb, S, infor


class InforExtractor(nn.Module):
    """Infor extractor. Builds node/time-dependent infor maps."""

    def __init__(self, N: int, d_model: int, num_hours: int = 24) -> None:
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.num_hours = num_hours
        self.spatial_vec1 = nn.Parameter(torch.randn(N, d_model))
        self.spatial_vec2 = nn.Parameter(torch.randn(N, d_model))
        self.temporal_vec1 = nn.Parameter(torch.randn(num_hours, d_model))
        self.temporal_vec2 = nn.Parameter(torch.randn(num_hours, d_model))

    def forward(
        self, x: torch.Tensor, tod_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        d = self.d_model
        # Per-node (d x d) map via outer products of two node vectors
        spatial_infor_mat = torch.einsum(
            'nd,ne->nde', self.spatial_vec1, self.spatial_vec2
        )  # (N, d, d)
        if tod_indices is not None:
            vec1_selected = self.temporal_vec1[tod_indices]  # (B, L, d)
            vec2_selected = self.temporal_vec2[tod_indices]  # (B, L, d)
            temporal_infor_mat = torch.einsum(
                'bld,ble->blde', vec1_selected, vec2_selected
            )  # (B, L, d, d)
        else:
            v1 = self.temporal_vec1[0:1]
            v2 = self.temporal_vec2[0:1]
            temporal_infor_mat = torch.einsum('ad,ae->ade', v1, v2).unsqueeze(0).expand(
                B, -1, -1, -1
            )
        pdb.set_trace()
        return spatial_infor_mat, temporal_infor_mat

    def extract_spatial_infor(
        self, x: torch.Tensor, infor_mat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B, N, L, d = x.shape
        x_reshaped = x.transpose(1, 2).contiguous().view(B * L, N, d)  # (B*L, N, d)
        infor = torch.einsum('bnd,nde->bne', x_reshaped, infor_mat)  # (B*L, N, d)
        return {'infor': infor, 'x': x_reshaped}

    def extract_temporal_infor(
        self,
        x: torch.Tensor,
        infor_mat: torch.Tensor,
        tod_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, L, d = x.shape
        x_reshaped = x.reshape(B * N, L, d)  # (B*N, L, d)
        pdb.set_trace()
        if tod_indices is not None and infor_mat.dim() == 4:
            infor_mat_expanded = infor_mat.unsqueeze(1).expand(-1, N, -1, -1, -1)
            infor_mat_expanded = infor_mat_expanded.reshape(B * N, L, d, d)  # (B*N, L, d, d)
            infor = torch.einsum('bld,blde->ble', x_reshaped, infor_mat_expanded)
        else:
            # Fallback: share a single temporal map across L
            shared = (
                infor_mat[0]
                if infor_mat.dim() == 3
                else torch.eye(d, device=x.device).unsqueeze(0).expand(L, d, d)
            )
            infor = torch.einsum('bld,ldk->blk', x_reshaped, shared)
        return {'infor': infor, 'x': x_reshaped}


# ===== Lightweight replacements for GNN/Attention =====
class SpatialMixer(nn.Module):
    def __init__(self, N: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.N = N
        self.A = nn.Parameter(torch.randn(N, N) * (1.0 / N))
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*L, N, d)
        A = torch.softmax(self.A, dim=-1)  # (N, N)
        y = torch.einsum('nm,bmd->bnd', A, x)  # simple learned mixing
        y = (y + x) * 0.5
        y = self.drop(self.proj(y))
        return y


class TemporalConvMixer(nn.Module):
    """Depthwise-separable 1D conv along time; no attention."""

    def __init__(
        self, d_model: int, kernel_size: int = 3, dropout: float = 0.1, causal: bool = False
    ) -> None:
        super().__init__()
        self.causal = causal
        padding = kernel_size - 1 if causal else (kernel_size // 2)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=padding)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, L, d)
        x_t = x.transpose(1, 2)  # (B*N, d, L)
        y = self.dw(x_t)
        if self.causal:
            y = y[:, :, : x_t.size(-1)]  # trim any future leakage
        y = self.pw(self.act(y))
        y = self.drop(y).transpose(1, 2)
        return x + y  # residual


class STIDBlock(nn.Module):
    """Position-wise MLP block (STID-style)."""

    def __init__(self, d_model: int, dropout: float = 0.1, norm_type: str = 'layer') -> None:
        super().__init__()
        self.pre_norm = NormChooser(d_model, norm_type)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_norm(x)
        return x + self.mlp(y)


import torch

def make_tod_indices_from_start_minutes(
    start_minute_of_day: torch.LongTensor,  # shape: (B,)
    L: int,                                 # window length
    freq_mins: int = 5,                     # 5 minutes → 288 slots/day
    slots_per_day: int = 288
) -> torch.LongTensor:
    """
    Returns (B, L) LongTensor with indices in [0, slots_per_day-1].
    start_minute_of_day is minutes since midnight for the first step in each sequence.
    """
    B = start_minute_of_day.shape[0]
    step = torch.arange(L, device=start_minute_of_day.device) * freq_mins  # (L,)
    mins = start_minute_of_day.unsqueeze(1) + step.unsqueeze(0)           # (B, L)
    mins_mod_day = mins % 1440
    return (mins_mod_day // (1440 // slots_per_day)).long()

# ===== Main: infor-aware STID =====
class AdaSTO(nn.Module):
    def __init__(
        self,
        N: int,
        L: int,
        L_pred: int,
        d_model: int = 64,
        num_layers: int = 3,
        input_dim: int = 1,
        output_dim: int = 1,
        num_hours: int = 24,
        norm_type: str = 'layer',
        dropout_p: float = 0.1,
        spatial_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 0,
        st_embedding_dim: int = 64,
        st_fuse_dropout: float = 0.1,
        # Mixer hyper-params
        temporal_kernel: int = 3,
        temporal_causal: bool = False,
    ) -> None:
        super().__init__()
        self.N, self.L, self.L_pred = N, L, L_pred
        self.d_model, self.num_layers, self.num_hours = d_model, num_layers, num_hours

        # Input projection
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.input_norm = NormChooser(d_model, norm_type=norm_type)

        # STID-style features
        self.spatial_embedding_dim = spatial_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.st_embedding_dim = st_embedding_dim

        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.empty(N, spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(num_hours, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if st_embedding_dim > 0:
            self.st_embedding = nn.Parameter(torch.empty(L, N, st_embedding_dim))
            nn.init.xavier_uniform_(self.st_embedding)

        added_dim = (
            spatial_embedding_dim + tod_embedding_dim + dow_embedding_dim + st_embedding_dim
        )
        self._use_st_features = added_dim > 0
        if self._use_st_features:
            self.st_fuse = nn.Sequential(
                nn.Linear(d_model + added_dim, d_model),
                nn.SiLU(),
                nn.Dropout(st_fuse_dropout),
            )

        # infor components (kept)
        self.infor_extractors = nn.ModuleList(
            [InforExtractor(N, d_model, num_hours) for _ in range(num_layers)]
        )
        self.infor_gatings = nn.ModuleList([SpatialTemporalGating(d_model) for _ in range(num_layers)])

        # Replacements for GNN/Attention
        self.infor_spatial_mixers = nn.ModuleList(
            [SpatialMixer(N, d_model, dropout=dropout_p) for _ in range(num_layers)]
        )
        self.infor_temporal_mixers = nn.ModuleList(
            [
                TemporalConvMixer(
                    d_model, kernel_size=temporal_kernel, dropout=dropout_p, causal=temporal_causal
                )
                for _ in range(num_layers)
            ]
        )

        # STID-like per-layer encoder = position-wise MLP
        self.blocks = nn.ModuleList(
            [STIDBlock(d_model, dropout=dropout_p, norm_type=norm_type) for _ in range(num_layers)]
        )

        # Heads
        self.temporal_projection = nn.Linear(L, L_pred)
        self.output_projection = nn.Linear(d_model, output_dim)

        self.component_attention = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),  # 3 components: emb, spatial, temporal
            nn.Softmax(dim=-1),
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        x = history_data
        t_i_d_data = history_data[..., 1]
        tod_indices = (t_i_d_data[..., -1] * 288).type(torch.LongTensor).to(x.device)
        B, L, N, I = x.shape
        assert N == self.N and L == self.L, (
            f"Input shapes do not match: got N={N},L={L}, expected {self.N},{self.L}"
        )
        x = x.transpose(1, 2)  # (B, N, L, I)
        # pdb.set_trace()

        # Base embedding + norm
        emb = self.input_embedding(x)  # (B, N, L, d)
        emb = self.input_norm(emb)

        # ST features (STID-style concat → fuse)
        if self._use_st_features:
            feats = [emb]
            if self.tod_embedding_dim > 0:
                tod_idx = tod_indices.unsqueeze(1).expand(-1, N, -1)
                feats.append(self.tod_embedding(tod_idx))
            if self.dow_embedding_dim > 0 and dow_indices is not None:
                dow_idx = dow_indices.unsqueeze(1).expand(-1, N, -1)
                feats.append(self.dow_embedding(dow_idx))
            if self.spatial_embedding_dim > 0:
                feats.append(self.node_emb.unsqueeze(0).unsqueeze(2).expand(B, N, L, -1))
            if self.st_embedding_dim > 0:
                feats.append(self.st_embedding.permute(1, 0, 2).unsqueeze(0).expand(B, N, L, -1))
            emb = self.st_fuse(torch.cat(feats, dim=-1))

        all_layer_gated_outputs: List[Dict[str, Dict[str, torch.Tensor]]] = []
        spatial_infor_modeled: Optional[torch.Tensor] = None
        temporal_infor_modeled: Optional[torch.Tensor] = None

        for layer_idx in range(len(self.blocks)):
            # 1) infor extraction
            spatial_infor_mat, temporal_infor_mat = self.infor_extractors[layer_idx](
                emb, tod_indices
            )
            spatial_infor_data = self.infor_extractors[layer_idx].extract_spatial_infor(
                emb, spatial_infor_mat
            )
            temporal_infor_data = self.infor_extractors[layer_idx].extract_temporal_infor(
                emb, temporal_infor_mat, tod_indices
            )

            # 3) Individual spatial/temporal infor modeling (no GNN / no attention)
            spatial_input = spatial_infor_data['infor'] + (
                spatial_infor_modeled if spatial_infor_modeled is not None else 0.0
            )
            spatial_infor_modeled = self.infor_spatial_mixers[layer_idx](spatial_input)

            temporal_input = temporal_infor_data['infor'] + (
                temporal_infor_modeled if temporal_infor_modeled is not None else 0.0
            )
            temporal_infor_modeled = self.infor_temporal_mixers[layer_idx](temporal_input)

            # 4) STID-style content path with infor gating
            #    Spatial gating (reshape to (B, N, L, d))
            spatial_infor_r = spatial_infor_modeled.view(B, L, N, self.d_model).transpose(1, 2)
            out_spatial, S_spatial, infor_spatial = self.infor_gatings[layer_idx].mlp_gating(
                emb, spatial_infor_r
            )
            emb = emb + out_spatial

            #    Temporal content mixer (Conv1d) + temporal gating
            x_t = emb.view(B * N, L, self.d_model)
            x_t = self.infor_temporal_mixers[layer_idx](x_t)  # reuse as content mixer
            x_t = x_t.view(B, N, L, self.d_model)
            temporal_infor_r = temporal_infor_modeled.view(B, N, L, self.d_model)
            out_temporal, S_temporal, infor_temporal = self.infor_gatings[
                layer_idx
            ].mlp_gating(x_t, temporal_infor_r)
            emb = emb + out_temporal

            all_layer_gated_outputs.append(
                {
                    'spatial': {
                        'out_emb': out_spatial,
                        'infor': infor_spatial,
                        'x': spatial_infor_data['x'],
                        'infor_x': spatial_infor_data['infor'],
                    },
                    'temporal': {
                        'out_emb': out_temporal,
                        'infor': infor_temporal,
                        'x': temporal_infor_data['x'],
                        'infor_x': temporal_infor_data['infor'],
                    },
                }
            )

            # 5) Position-wise MLP block (STID-style)
            emb = self.blocks[layer_idx](emb)

        # 6) Heads: project over time & aggregate last-layer infors
        emb_projected = self.temporal_projection(emb.transpose(2, 3)).transpose(
            2, 3
        )  # (B, N, L_pred, d)

        if all_layer_gated_outputs:
            last_gated = all_layer_gated_outputs[-1]
            spatial_infor_proj = self.temporal_projection(
                last_gated['spatial']['infor'].transpose(2, 3)
            ).transpose(2, 3)
            temporal_infor_proj = self.temporal_projection(
                last_gated['temporal']['infor'].transpose(2, 3)
            ).transpose(2, 3)

            # Component attention
            combined_features = torch.cat(
                [emb_projected, spatial_infor_proj, temporal_infor_proj], dim=-1
            )  # (B, N, L_pred, 3*d)
            attention_weights = self.component_attention(combined_features)  # (B, N, L_pred, 3)

            # Project each component then fuse
            emb_output = self.output_projection(emb_projected)
            spatial_output = self.output_projection(spatial_infor_proj)
            temporal_output = self.output_projection(temporal_infor_proj)

            stacked_outputs = torch.stack(
                [emb_output, spatial_output, temporal_output], dim=-1
            )  # (B, N, L_pred, D_out, 3)
            final_output = (stacked_outputs * attention_weights.unsqueeze(-2)).sum(
                dim=-1
            )  # (B, N, L_pred, D_out)

            final_output = final_output.transpose(1, 2)  # (B, L_pred, N, D_out)

            component_importance = attention_weights.mean(dim=[0, 1, 2])  # (3,)

            return final_output
            # , all_layer_gated_outputs, {
            #     'attention_weights': attention_weights.transpose(1, 2),  # (B, L_pred, N, 3)
            #     'component_importance': component_importance,  # (3,) [emb, spatial, temporal]
            #     'spatial_contribution': spatial_output.transpose(1, 2),
            #     'temporal_contribution': temporal_output.transpose(1, 2),
            # }
        else:
            output = self.output_projection(emb_projected)
            return output.transpose(1, 2)
            # , all_layer_gated_outputs, None

    # ===== Orthogonal & Sparse losses (kept) =====
    def orthogonal_and_sparse_losses(
        self, lambda_orth: float = 1e-3, lambda_sparse: float = 1e-6
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        # Orthogonality on embeddings
        orth_loss = 0.0
        mats = []
        if hasattr(self, 'node_emb'):
            mats.append(self.node_emb)
        if hasattr(self, 'tod_embedding'):
            mats.append(self.tod_embedding.weight)
        if hasattr(self, 'dow_embedding'):
            mats.append(self.dow_embedding.weight)
        if hasattr(self, 'st_embedding'):
            mats.append(self.st_embedding.view(-1, self.st_embedding.shape[-1]))
        for M in mats:
            M_ = M - M.mean(dim=0, keepdim=True)
            M_ = F.normalize(M_, dim=1)
            gram = M_.t() @ M_
            I = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
            orth_loss = orth_loss + (gram - I).pow(2).mean()
        losses['orthogonal_loss'] = lambda_orth * orth_loss

        # Sparsity on infor/mixer parameters
        sparse_loss = 0.0
        for mod in list(self.infor_extractors) + list(self.infor_spatial_mixers) + list(
            self.infor_temporal_mixers
        ):
            for p in mod.parameters():
                sparse_loss = sparse_loss + p.abs().mean()
        losses['sparse_loss'] = lambda_sparse * sparse_loss
        losses['aux_loss'] = losses['orthogonal_loss'] + losses['sparse_loss']
        return losses
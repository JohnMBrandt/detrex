import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


from detrex.layers import (
    MultiScaleDeformableAttention,
)


#from detectron2.projects.deformable_detr import MSDeformAttn
from timm.models.layers import DropPath  # or import from timm if preferred

_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, device=device)
        )
        ref = torch.stack((ref_x.reshape(-1) / W_, ref_y.reshape(-1) / H_), -1)
        reference_points_list.append(ref[None])
    reference_points = torch.cat(reference_points_list, 1)  # [1, sum(H*W), 2]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    device = x.device

    # --- 1) Injector inputs: query length = patch‐tokens (h//16 * w//16)
    # We want reference_points1 shape [bs, Nq, 3, 2]
    # where Nq = (h//16)*(w//16), and n_levels=3

    # 1a) spatial_shapes for value (unused by RP here, but needed downstream)
    spatial_shapes1 = torch.as_tensor([
        (h // 8,  w // 8),
        (h // 16, w // 16),
        (h // 32, w // 32),
    ], dtype=torch.long, device=device)
    level_start_index1 = torch.cat((
        spatial_shapes1.new_zeros((1,)),
        spatial_shapes1.prod(1).cumsum(0)[:-1],
    ))

    # 1b) base reference for the patch‐level (h//16, w//16)
    H_patch, W_patch = h // 16, w // 16
    rp16 = get_reference_points([(H_patch, W_patch)], device)  # [1, Nq, 2]
    # replicate across 3 levels → [1, Nq, 3, 2]
    rp1 = rp16.unsqueeze(2).repeat(1, 1, 3, 1)

    deform1 = [rp1, spatial_shapes1, level_start_index1]

    # --- 2) Extractor inputs: value length = sum(spatial_shapes1)
    # We want reference_points2 shape [bs, Nv, 1, 2] with n_levels=1

    # 2a) we’ll treat only the “middle” level for segmenting value (like original)
    spatial_shapes2 = torch.as_tensor([
        (h // 16, w // 16),
    ], dtype=torch.long, device=device)
    level_start_index2 = torch.zeros((1,), dtype=torch.long, device=device)

    # 2b) build a flat reference over all 3 levels, then collapse to 1-level
    rp_all = get_reference_points([
        (h // 8,  w // 8),
        (h // 16, w // 16),
        (h // 32, w // 32),
    ], device)  # [1, Nall, 2]
    rp2 = rp_all.unsqueeze(2)  # → [1, Nall, 1, 2]

    deform2 = [rp2, spatial_shapes2, level_start_index2]

    return deform1, deform2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Injector(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=3,
        init_values=0.,
        with_cp=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm  = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_levels=n_levels,
            num_points=n_points,
            img2col_step=64,
            dropout=0.0,
            batch_first=True,
        )
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, query, reference_points, feat, spatial_shapes, lvl_start):
        def _inner(q, f):
            attn_out = self.attn(
                query              = self.query_norm(q),
                value              = self.feat_norm(f),
                reference_points   = reference_points,
                spatial_shapes     = spatial_shapes,
                level_start_index  = lvl_start,
            )
            return q + self.gamma * attn_out

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner, query, feat, use_reentrant = False)
        else:
            return _inner(query, feat)


class Extractor(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        n_levels=1,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.,
        drop_path=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        with_cp=False,
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm  = norm_layer(dim)
        self.attn = MultiScaleDeformableAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_levels=n_levels,
            num_points=n_points,
            img2col_step=64,
            dropout=drop,
            batch_first=True,
        )
        self.with_cffn = with_cffn
        self.with_cp   = with_cp
        if with_cffn:
            self.ffn = ConvFFN(dim, int(dim * cffn_ratio), dim, drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, lvl_start, H, W):
        def _inner(q, f):
            attn_out = self.attn(
                query              = self.query_norm(q),
                value              = self.feat_norm(f),
                reference_points   = reference_points,
                spatial_shapes     = spatial_shapes,
                level_start_index  = lvl_start,
            )
            q = q + attn_out
            if self.with_cffn:
                q = q + self.drop_path(self.ffn(self.ffn_norm(q), H, W))
            return q

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner, query, feat, use_reentrant = False)
        else:
            return _inner(query, feat)

class InteractionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        norm_layer=nn.LayerNorm,
        drop: float = 0.0,
        drop_path: float = 0.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        init_values: float = 0.0,
        extra_extractor: bool = False,
        with_cp: bool = False,
    ):
        super().__init__()
        # Injector will attend over levels=3 (8×,16×,32× tokens)
        self.injector = Injector(
            dim=dim,
            num_heads=num_heads,
            n_points=n_points,
            n_levels=3,
            init_values=init_values,
            with_cp=with_cp,
            norm_layer=norm_layer,
        )
        # Extractor will attend over levels=1 (just the fused tokens)
        self.extractor = Extractor(
            dim=dim,
            num_heads=num_heads,
            n_points=n_points,
            n_levels=1,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            with_cp=with_cp,
        )
        # Optionally add two more extractors in series
        if extra_extractor:
            self.extra_extractors = nn.ModuleList([
                Extractor(
                    dim=dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    n_levels=1,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    drop=drop,
                    drop_path=drop_path,
                    norm_layer=norm_layer,
                    with_cp=with_cp,
                )
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        blocks: list[nn.Module],
        deform_inputs1: list[torch.Tensor],
        deform_inputs2: list[torch.Tensor],
        H: int,
        W: int,
    ):
        # 1) Injector: mix spatial priors into x
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            lvl_start=deform_inputs1[2],
        )

        # 2) Run the specified ViT blocks on x
        for blk in blocks:
            x = blk(x)#, H, W)

        # 3) Extractor: deform‐attention from x back into c
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            lvl_start=deform_inputs2[2],
            H=H,
            W=W,
        )

        # 4) Extra extractors if requested
        if self.extra_extractors is not None:
            for ext in self.extra_extractors:
                c = ext(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    lvl_start=deform_inputs2[2],
                    H=H,
                    W=W,
                )

        return x, c

import torch
import torch.nn as nn
from detectron2.layers import Conv2d  # or just use nn.Conv2d

class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, num_groups=32):
        super().__init__()

        # convenience
        GN = lambda channels: nn.GroupNorm(num_groups, channels)

        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            GN(inplanes),
            nn.ReLU(inplace=True),

            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            GN(inplanes),
            nn.ReLU(inplace=True),

            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            GN(inplanes),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            GN(2 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            GN(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            GN(4 * inplanes),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Conv2d(inplanes,     embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2*inplanes,   embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4*inplanes,   embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4*inplanes,   embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        # project each to embed_dim
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        B, D, _, _ = c1.shape
        # reshape into token sequences
        # c1 is stride-4 → you may or may not need it; original code dropped it
        #c1 = c1.view(B, D, -1).transpose(1, 2)

        c2 = c2.view(B, D, -1).transpose(1, 2)  # stride-8
        c3 = c3.view(B, D, -1).transpose(1, 2)  # stride-16
        c4 = c4.view(B, D, -1).transpose(1, 2)  # stride-32

        return c1, c2, c3, c4
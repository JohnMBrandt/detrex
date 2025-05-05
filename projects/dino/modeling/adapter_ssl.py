import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec, get_norm, Conv2d, ConvTranspose2d
from detrex.layers import MultiScaleDeformableAttention


from detectron2.modeling.backbone.SSLVisionTransformer import SSLVisionTransformer
#from projects.ssl_vit.ssl_vit_backbone import SSLVisionTransformer
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

@BACKBONE_REGISTRY.register()
class ViTAdapterSSL(SSLVisionTransformer):
    """
    A ViTAdapter that *is* an SSLVisionTransformer under the hood,
    so your DiNOv2 pretrained ViT weights load 1:1, then adds adapter layers.
    """
    def __init__(
        self,
        # ==== SSLViTBackbone args ====
        img_size: int            = 512,
        patch_size: int          = 16,
        embed_dim: int           = 384,
        depth: int               = 24,
        num_heads: int           = 16,
        mlp_ratio: float         = 4.0,
        qkv_bias: bool           = True,
        drop_path_rate: float    = 0.0,
        out_indices: tuple       = [4, 11, 17, 23],
        final_norm: bool         = False,
        with_cls_token: bool     = True,
        output_cls_token: bool   = False,
        frozen_stages: int       = 100,
        fpn: bool                = False,
        init_cfg=None,
        pretrained: str          = None,
        # ==== Adapter-specific args ====
        pretrain_size: int       = 224,
        conv_inplane: int        = 64,
        n_points: int            = 4,
        deform_num_heads: int    = 6,
        init_values: float       = 0.0,
        interaction_indexes: list= (4, 11, 17, 23),
        with_cffn: bool          = True,
        cffn_ratio: float        = 0.25,
        add_vit_feature: bool    = True,
        use_extra_extractor: bool= True,
        num_groups: int          = 32,
    ):
        # 1) Initialize the exact SSLViT you pretrained
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
            final_norm=final_norm,
            with_cls_token=with_cls_token,
            output_cls_token=output_cls_token,
            frozen_stages=frozen_stages,
            fpn=fpn,
            init_cfg=init_cfg,
            pretrained=None,  # load manually below
        )

        # 2) Load DiNOv2 checkpoint into SSLViT base
        if pretrained:
            ckpt = torch.load(pretrained, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            # strip common prefixes
            state = {k.replace("vit.", ""): v for k, v in state.items()}
            state = {k.replace("model.", ""): v for k, v in state.items()}

            own = self.state_dict()
            to_load = {k: v for k, v in state.items()
                       if k in own and v.shape == own[k].shape}
            self.load_state_dict(to_load, strict=False)
            print(f"[ViTAdapterSSL] loaded {len(to_load)}/{len(own)} SSL weights from {pretrained}")
            
        # 3) Adapter-specific setup
        self.pretrain_size       = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature     = add_vit_feature
        D = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, D))
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane,
            embed_dim=D,
            num_groups=num_groups,
        )

        self.interactions = nn.Sequential(*[
            InteractionBlock(
                dim=D,
                num_heads=deform_num_heads,
                n_points=n_points,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop=0.0,
                drop_path=drop_path_rate,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                init_values=init_values,
                extra_extractor=(i == len(interaction_indexes) - 1 and use_extra_extractor),
                with_cp=True,
            )
            for i in range(len(interaction_indexes))
        ])

        self.up    = ConvTranspose2d(D, D, kernel_size=2, stride=2)
        self.norm1 = get_norm("GN", D)
        self.norm2 = get_norm("GN", D)
        self.norm3 = get_norm("GN", D)
        self.norm4 = get_norm("GN", D)

        # init adapter weights only
        self._init_weights_adapter()
        self.init_weights()

        # 4) Register output feature metadata for Detectron2
        strides  = [patch_size // 4, patch_size // 2, patch_size, patch_size * 2]
        names    = [f"p{int(math.log2(s))}" for s in strides]
        channels = [D] * 4
        self._out_features         = names
        self._out_feature_strides  = dict(zip(names, strides))
        self._out_feature_channels = dict(zip(names, channels))

    def _init_weights_adapter(self):
        # Copy/paste of MMDet ViTAdapter's _init_weights
        def _init(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        self.spm.apply(_init)
        self.interactions.apply(_init)
        self.up.apply(_init)
        normal_(self.level_embed)

        # Reset any MSDeformAttn layers
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def _get_pos_embed(self, pos_embed, H, W):
        # Bicubic resizing of the pretrained pos_emb
        pe = pos_embed.reshape(1, self.pretrain_size[0] // 16,
                               self.pretrain_size[1] // 16, -1)
        pe = pe.permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(H, W), mode='bicubic', align_corners=False)
        pe = pe.reshape(1, -1, H * W).permute(0, 2, 1)
        return pe

    def forward(self, x):
        # 1) Deform inputs & spatial priors
        deform1, deform2 = deform_inputs(x)
        c1, c2, c3, c4 = self.spm(x)
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]

        B, C, H_img, W_img = x.shape
        bs = B
        x, _, _ = self.prepare_tokens(x)     # returns [B, 1+H*W, D] if with_cls_token=True
    
        # --- **DROP** the leading cls token so we have exactly H*W tokens ---
        if self.with_cls_token:
            x = x[:, 1:, :]
    
        # recompute H and W from the image / patch size
        patch = (self.patch_size if isinstance(self.patch_size, int)
                 else self.patch_size[0])
        H = H_img // patch
        W = W_img // patch
    
        # now x is [B, H*W, D], same as pos after resizing
        #print(self.pos_embed.shape)
        pos = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        #print(pos.shape)
        x = self.pos_drop(x + pos)

        # 3) Interaction blocks
        c = torch.cat([c2, c3, c4], dim=1)
        for i, block in enumerate(self.interactions):
            idxs = self.interaction_indexes[i]
            x, c = block(x, c, self.blocks[idxs[0]:idxs[-1] + 1],
                         deform1, deform2, H, W)

        # 4) Split token stream
        n2, n3 = c2.size(1), c3.size(1)
        c2_tok = c[:, :n2, :]
        c3_tok = c[:, n2:n2 + n3, :]
        c4_tok = c[:, n2 + n3:, :]

        # 5) Reshape to feature maps
        c2_feat = c2_tok.transpose(1, 2).view(bs, self.embed_dim, H * 2, W * 2)
        c3_feat = c3_tok.transpose(1, 2).view(bs, self.embed_dim, H, W)
        c4_feat = c4_tok.transpose(1, 2).view(bs, self.embed_dim, H // 2, W // 2)
        c1_feat = self.up(c2_feat) + c1

        # 6) Optional ViT feature fusion
        if self.add_vit_feature:
            x_map = x.transpose(1, 2).view(bs, self.embed_dim, H, W)
            x1 = F.interpolate(x_map, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x_map, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x_map, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1_feat = c1_feat + x1
            c2_feat = c2_feat + x2
            c3_feat = c3_feat + x_map
            c4_feat = c4_feat + x4

        # 7) Normalize
        f1 = self.norm1(c1_feat)
        f2 = self.norm2(c2_feat)
        f3 = self.norm3(c3_feat)
        f4 = self.norm4(c4_feat)
        # 8) Return multi-scale dict
        return {n: t for n, t in zip(self._out_features, [f1, f2, f3, f4])}

    def output_shape(self):
        return {name: ShapeSpec(channels=self._out_feature_channels[name],
                                stride=self._out_feature_strides[name])
                for name in self._out_features}
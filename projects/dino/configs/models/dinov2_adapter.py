from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.SSLVisionTransformer import SSLViTBackbone

from .dino_r50 import model
from ...modeling.adapter_ssl import ViTAdapterSSL

model.backbone = L(ViTAdapterSSL)(
    # ==== SSLViT args ====
    img_size           = 512,
    patch_size         = 16,
    embed_dim          = 1280,
    depth              = 32,
    num_heads          = 20,
    mlp_ratio          = 4.0,
    qkv_bias           = True,
    drop_path_rate     = 0.0,
    out_indices        = [5, 14, 22, 31],
    final_norm         = False,
    with_cls_token     = True,
    output_cls_token   = False,
    frozen_stages      = 100,
    pretrained = '/home/ubuntu/mmdetection/models/SSLhuge_satellite.pth',
    fpn                = False,   # disable the small FPN in the SSL class
    init_cfg           = dict(
      type       = "Pretrained",
      checkpoint = "/home/ubuntu/mmdetection/models/SSLhuge_satellite.pth"
    ),
    # ==== Adapter args ====
    pretrain_size        = 512,            # original pretrain resolution
    conv_inplane         = 64,             # conv‐stem width
    n_points             = 4,              # deform sample points
    deform_num_heads     = 4,              # heads in deform‐attn
    init_values          = 0.0,            # small initial gamma in injector
    interaction_indexes = [
  [ 0,  7],   # run blocks 0→7, then adapter #1
  [ 8, 15],   # run blocks 8→15, then adapter #2
  [16, 23],   # run blocks 16→23, then adapter #3
  [24, 31],   # run blocks 24→31, then adapter #4
],#[[4, 5], [13, 14], [21, 22], [30, 31]],  # line up with out_indices
    with_cffn            = True,
    cffn_ratio           = 0.25,
    #deform_ratio         = 1.0,
    add_vit_feature      = True,
    use_extra_extractor  = True,
    num_groups           = 32,             # for GroupNorm in the adapter
)
# modify neck config
model.neck.input_shapes = {
    "p2": ShapeSpec(channels=1280),
    "p3": ShapeSpec(channels=1280),
    "p4": ShapeSpec(channels=1280),
    "p5": ShapeSpec(channels=1280),
}
model.neck.in_features = ["p2", "p3", "p4", "p5"]
model.neck.num_outs = 4
model.transformer.num_feature_levels = 4
from .deformable_detr_r50 import model
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec

from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.SSLVisionTransformer import SSLViTBackbone

# modify model config
model.with_box_refine = True
model.as_two_stage = True

# modify training config
#train.output_dir = "./output/deformable_detr_r50_two_stage_12ep"

model.transformer.encoder.feedforward_dim=2048
model.transformer.decoder.feedforward_dim=2048

model.transformer.encoder.attn_dropout=0.0
model.transformer.encoder.ffn_dropout=0.0
model.transformer.decoder.attn_dropout=0.0
model.transformer.decoder.ffn_dropout=0.0

model.mixed_selection = True
model.transformer.mixed_selection = True
model.transformer.decoder.look_forward_twice = True


#model.select_box_nums_for_evaluation = 300
model.num_queries = 1750
model.transformer.encoder.use_checkpoint=True

# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SSLViTBackbone)(
    # MM-Detection parameters:
    img_size=512,            # must be defined elsewhere
    patch_size=16,
    embed_dim=1280,
    depth=32,
    num_heads=20,
    mlp_ratio=4,
    qkv_bias=True,
    drop_path_rate=0.0,
    frozen_stages=100,
    pretrained=None,
    out_indices=[5, 14, 22, 31],
    init_cfg=dict(
        type="Pretrained",
        checkpoint="/home/ubuntu/mmdetection/models/SSLhuge_satellite.pth",
    ),
)

# modify neck config
model.neck.input_shapes = {
    "p2": ShapeSpec(channels=1280 // 4),
    "p3": ShapeSpec(channels=1280 // 2),
    "p4": ShapeSpec(channels=1280),
    "p5": ShapeSpec(channels=1280),
}
model.neck.in_features = ["p2", "p3", "p4", "p5"]
model.neck.num_outs = 4
model.transformer.num_feature_levels = 4
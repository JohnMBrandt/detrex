import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from projects.stable_dino.modeling import (
    DINO,
    StableDINOTransformerEncoder,
    DINOTransformerDecoder,
    DINOTransformer,
    StableDINOCriterion,
    StableDINOHungarianMatcher
)

model = L(DINO)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res3", "res4", "res5"],
        freeze_at=1,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=20,
        normalize=True,
        offset=0,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "res3": ShapeSpec(channels=512),
            "res4": ShapeSpec(channels=1024),
            "res5": ShapeSpec(channels=2048),
        },
        in_features=["res3", "res4", "res5"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(DINOTransformer)(
        encoder=L(StableDINOTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
            multi_level_fusion="dense-fusion"
        ),
        decoder=L(DINOTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
        ),
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
    ),
    embed_dim=256,
    num_classes=1,
    num_queries=1500,
    aux_loss=True,
    criterion=L(StableDINOCriterion)(
        num_classes="${..num_classes}",
        matcher=L(StableDINOHungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
            cec_beta=0.5,
        ),
        weight_dict={
            "loss_class": 6.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
        use_ce_loss_type="stable-dino",
        ta_alpha=0.0,
        ta_beta=2.0,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean = [107.2, 104.8, 75.38],
    pixel_std = [54.21, 39.81, 36.52],
    #pixel_mean=[123.675, 116.280, 103.530],
    #pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
    gdn_k=2,
    neg_step_type='none',
    no_img_padding=False,
    dn_to_matching_block=False,
)
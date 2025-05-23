import torch.nn as nn

from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.layers import ShapeSpec
from detectron2.config import LazyCall as L

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from projects.mr_detr_dino.modeling import (
    DeformableDETR,
    DeformableDetrTransformerEncoder,
    DeformableDetrTransformerDecoder,
    DeformableDetrTransformer,
    DeformableCriterion,
    DNCriterion
)
model = L(DeformableDETR)(
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
    transformer=L(DeformableDetrTransformer)(
        encoder=L(DeformableDetrTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(DeformableDetrTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
        ),
        as_two_stage="${..as_two_stage}",
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
        mixed_selection=False,
    ),
    embed_dim=256,
    num_classes=1,
    num_queries=300,
    aux_loss=True,
    with_box_refine=False,
    as_two_stage=False,
    criterion=L(DNCriterion)(
        num_classes=1,
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        enc_matcher=L(HungarianMatcher)(
            cost_class=0.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 2.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 2.0,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
    ),
    pixel_mean = [107.2, 104.8, 75.38],
    pixel_std = [54.21, 39.81, 36.52],
    #pixel_mean=[123.675, 116.280, 103.530],
    #pixel_std=[58.395, 57.120, 57.375],
    #select_box_nums_for_evaluation=300,
    device="cuda",
    mixed_selection=False, # tricks
    dn_label_noise_ratio=0.5,
    dn_box_noise_scale=0.4
)


weight_dict_enc={
     "loss_class": 2.0,
     "loss_bbox": 5.0,
     "loss_giou": 2.0
}
weight_dict_enc_o2m={
     "loss_class": 2.0,
     "loss_bbox": 5.0,
     "loss_giou": 2.0
}
weight_dict_o2m={
     "loss_class": 2.0,
     "loss_bbox": 5.0,
     "loss_giou": 2.0,
     "loss_class_dn": 2.0,
     "loss_bbox_dn": 5.0,
     "loss_giou_dn": 2.0,
}
weight_dict_sep={
     "loss_class": 2.0,
     "loss_bbox": 5.0,
     "loss_giou": 2.0,
     "loss_class_dn": 2.0,
     "loss_bbox_dn": 5.0,
     "loss_giou_dn": 2.0,
}
#weight_dict_enc = model.criterion.weight_dict


supp = {
    "loss_class_dn_o2m": 2.0,
    "loss_bbox_dn_o2m": 5.0,
    "loss_giou_dn_o2m": 2.0,
    "loss_class_dn_sep": 2.0,
    "loss_bbox_dn_sep": 5.0,
    "loss_giou_dn_sep": 2.0,
    "loss_class_dn_group": 2.0,
    "loss_bbox_dn_group": 5.0,
    "loss_giou_dn_group": 2.0,
}

if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    aux_weight_dict.update({k + "_enc": v for k, v in weight_dict_enc.items()})
    aux_weight_dict.update({k + "_enc_o2m": v for k, v in weight_dict_enc_o2m.items()})
    aux_weight_dict.update({k + "_enc_sep": v for k, v in weight_dict_enc_o2m.items()})
    for i in range(6):
        aux_weight_dict.update({k + f"_group_{i}": v for k, v in weight_dict_o2m.items()})
    for i in range(6):
        aux_weight_dict.update({k + f"_o2m_{i}": v for k, v in weight_dict_o2m.items()})
    for i in range(6):
        aux_weight_dict.update({k + f"_sep_{i}": v for k, v in weight_dict_sep.items()})
    weight_dict.update(aux_weight_dict)
    weight_dict.update(supp)
    model.criterion.weight_dict = weight_dict
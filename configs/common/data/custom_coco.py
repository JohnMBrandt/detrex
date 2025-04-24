# custom_coco.py
from omegaconf import OmegaConf
from detectron2.config import LazyCall as L
from detectron2.data.datasets import register_coco_instances
import detectron2.data.transforms as T
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detrex.data import DetrDatasetMapper

# ─── 1) REGISTER YOUR DATASETS ───────────────────────────────────────────
# Replace these JSON & image‐folder paths with your own:
register_coco_instances(
    "my_dataset_train", {}, 
    "/home/ubuntu/mmdetection/data/train.json", 
    "/home/ubuntu/mmdetection/data/train"
)
register_coco_instances(
    "my_dataset_val", {}, 
    "/home/ubuntu/mmdetection/data/val.json", 
    "/home/ubuntu/mmdetection/data/val"
)

# ─── 2) SCALE + CROP PARAMS ─────────────────────────────────────────────
MIN_SIZE = int(round(0.67 * 512))  # ≃342
MAX_SIZE = int(round(1.5  * 512))  # 768
dataloader = OmegaConf.create()

# training loader
dataloader.train = L(build_detection_train_loader)(
    # tell detectron2 to load your registered splits
    dataset=L(get_detection_dataset_dicts)(names="my_dataset_train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            # 1) random horizontal flip
            L(T.RandomFlip)(),
            # 2) random resize: shortest edge sampled uniformly between 0.67*512 and 1.5*512
            L(T.ResizeShortestEdge)(
                short_edge_length=(int(512 * 0.67), int(512 * 1.5)),
                max_size=int(512 * 1.5),
                sample_style="range",  # uniformly in the range
            ),
            # 3) then a fixed-size random crop to exactly 512×512
            L(T.RandomCrop)(
                crop_type="absolute",
                crop_size=(512, 512),
            ),
        ],
        augmentation_with_crop=None,  # we already do our own crop above
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=4,   # adjust to your GPUs / batch size
    num_workers=4,
)

# validation / test loader
dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="my_dataset_val",
        filter_empty=False,
    ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            # for inference, just resize so the short edge is 512 and max 512
            L(T.ResizeShortestEdge)(
                short_edge_length=512,
                max_size=512,
            ),
            # if for some reason you still want center-crop, see note below
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

# COCO evaluator
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    output_dir="./output"  # or wherever you want preds dumped
)
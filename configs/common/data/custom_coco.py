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
from detectron2.data.transforms.augmentation_impl import FixedSizeCrop
from ......my_transforms.albu_wrapper import AlbumentationsWrapper


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

from detectron2.config import LazyCall as L
from detectron2.data import transforms as T
from detectron2.data.transforms import (
    Transform,
    TransformGen,
    FixedSizeCrop,          # you use this later
)

import albumentations as A
import numpy as np
from omegaconf import OmegaConf

# --- 2) ALBUMENTATIONS ↔ DETECTRON2 BRIDGE -----------------------------------
class AlbumentationsTransform(Transform):
    """
    A Detectron2 Transform that applies a *deterministic* Albumentations
    augmentation to an image and leaves geometric data unchanged.
    """
    def __init__(self, albu_aug: A.BasicTransform, image: np.ndarray):
        super().__init__()
        # run once, cache result so further calls are deterministic
        self._out = albu_aug(image=image)

    # Detectron2 will call these:
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        # we ignore the supplied img because we already transformed `image`
        return self._out["image"]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class AlbumentationsWrapper(TransformGen):
    """
    Wrap **any** Albumentations transform (class + kwargs) as a TransformGen.
    Example:
        L(AlbumentationsWrapper)(aug=A.HorizontalFlip, p=0.5)
    """
    def __init__(self, *, aug: type, **aug_kwargs):
        """
        Parameters
        ----------
        aug : type
            Albumentations transform **class** (e.g. `A.RandomFog`, `A.OneOf`)
        **aug_kwargs :
            Parameters forwarded to `aug(...)` when it is instantiated.
        """
        super().__init__()
        self._aug_cls = aug
        self._aug_kwargs = aug_kwargs

    # Detectron2 calls this once per sample to get a *deterministic* transform
    def get_transform(self, image: np.ndarray) -> Transform:
        albu_aug = self._aug_cls(**self._aug_kwargs)
        return AlbumentationsTransform(albu_aug, image)


# Optional helper if you ever need the LazyCall factory
def Albu(*, aug: type, **aug_kwargs):
    return AlbumentationsWrapper(aug=aug, **aug_kwargs)


# --- 3) DATALOADER STANZA ----------------------------------------------------
# constants reused later
IMG_SZ = 512
MIN_SIZE = int(round(0.67 * IMG_SZ))
MAX_SIZE = int(round(1.50 * IMG_SZ))

dataloader = OmegaConf.create()
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="my_dataset_train",
        filter_empty=False
    ),
    mapper=L(DetrDatasetMapper)(
        # ---------------------------------------------------------------------
        #  A U G M E N T A T I O N  P I P E L I N E
        # ---------------------------------------------------------------------
        augmentation=[
            # ---- group of fog / defocus / brightness combos -----------------
            L(AlbumentationsWrapper)(
                aug=A.OneOf,
                transforms=[
                    A.Sequential([
                        A.RandomFog(p=0.5, fog_coef_lower=0.15,
                                    fog_coef_upper=0.8, alpha_coef=0.1),
                        A.Defocus(p=0.5, radius=(2, 5),
                                  alias_blur=(0.1, 0.5)),
                        A.RandomBrightnessContrast(
                            brightness_limit=(.05, .25),
                            contrast_limit=(-.25, -.05), p=0.5),
                    ], p=1.0),
                    A.Sequential([
                        A.RandomFog(p=0.3, fog_coef_lower=0.1,
                                    fog_coef_upper=0.7, alpha_coef=0.1),
                        A.Defocus(p=0.2, radius=(2, 4),
                                  alias_blur=(0.1, 0.4)),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-.2, .2),
                            contrast_limit=(-.2, .2), p=1.0),
                    ], p=1.0),
                    A.Sequential([
                        A.Defocus(p=0.3, radius=(2, 3),
                                  alias_blur=(0.1, 0.5)),
                        A.RandomGamma(gamma_limit=(150, 195), p=0.1),
                    ], p=1.0),
                ],
                p=0.2,
            ),

            # ---- HSV shift ---------------------------------------------------
            L(AlbumentationsWrapper)(
                aug=A.HueSaturationValue,
                hue_shift_limit=8,
                sat_shift_limit=12,
                val_shift_limit=12,
                p=0.2,
            ),

            # ---- Gaussian noise ---------------------------------------------
            L(AlbumentationsWrapper)(
                aug=A.GaussNoise,
                var_limit=(10, 200),
                mean=0.0,
                p=0.2,
            ),

            # ---- crop → resize → crop → flips (Detectron2 transforms) -------
            L(FixedSizeCrop)(
                crop_size=(IMG_SZ, IMG_SZ),
                pad=True, pad_value=0.0, seg_pad_value=255
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(MIN_SIZE, MAX_SIZE),
                max_size=MAX_SIZE,
                sample_style="range",
            ),
            L(FixedSizeCrop)(
                crop_size=(IMG_SZ, IMG_SZ),
                pad=True, pad_value=0.0, seg_pad_value=255
            ),
            L(T.RandomFlip)(),  # prob defaults to 0.5, flip direction random
        ],
        augmentation_with_crop=None,    # we took care of cropping ourselves
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=4,
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
    output_dir="./output",  # or wherever you want preds dumped
    max_dets_per_image=1000,
)
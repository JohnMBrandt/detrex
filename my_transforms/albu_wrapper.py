from detectron2.data.transforms import Transform, TransformGen
from detectron2.data import transforms as T
import albumentations as A
import numpy as np

__all__ = ["AlbumentationsWrapper"]       # <- helps static-analysis tools


class AlbumentationsTransform(Transform):
    """Apply a *deterministic* Albumentations augmentation to an image only."""
    def __init__(self, albu_aug: A.BasicTransform, image: np.ndarray):
        super().__init__()
        self._out = albu_aug(image=image)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return self._out["image"]

    def apply_coords(self, coords):
        return coords


class AlbumentationsWrapper(TransformGen):
    """
    Turn **any** Albumentations transform class into a Detectron2 `TransformGen`.

    Example in a LazyCall/YAML config:
        _target_: my_transforms.albu_wrapper.AlbumentationsWrapper
        aug: !!python/name:albumentations.augmentations.transforms.HueSaturationValue ''
        p: 0.2
        hue_shift_limit: 8
        ...
    """
    def __init__(self, *, aug: type, **aug_kwargs):
        super().__init__()
        self._aug_cls   = aug
        self._aug_kw    = aug_kwargs

    def get_transform(self, image):
        return AlbumentationsTransform(self._aug_cls(**self._aug_kw), image)
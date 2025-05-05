import cv2
import torch
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import Resize
from detectron2.utils.visualizer import Visualizer

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.utils.visualizer import Visualizer, ColorMode
import torchvision


# 1) Load your detrex config as a LazyConfig
cfg = LazyConfig.load(
    "/home/ubuntu/arizona-test/detrex/projects/dino/configs/dino-vitdet/dino_dinov2-adapt.py"
)
# override the weights and device if you like
cfg.train.init_checkpoint = "/home/ubuntu/arizona-test/detrex/projects/dino/output/dino_vithuge-512-augment-2048-adapt-pretrain-fix/model_0119999.pth"
cfg.model.device = "cuda"  # or "cpu"

# 2) Instantiate the model
model = instantiate(cfg.model)
DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
model.to(cfg.model.device).eval()

# 3) Read and preprocess your image
img = cv2.imread("/home/ubuntu/mmdetection/data/train/59d25c84-311.png")
#img = cv2.imread("/home/ubuntu/mmdetection/data/train/58c46626-1024.0_36864.0.png")
#img = cv2.imread("/home/ubuntu/mmdetection/data/train/54f72b03-146.png")
#img = cv2.imread("/home/ubuntu/mmdetection/data/train/39e6be96-oilpalm-guatemala-image2-2048_4608.png")
#img = cv2.imread("/home/ubuntu/mmdetection/data/train/b1903af8-iggs_clip_1280_512.png")
img = cv2.imread("/home/ubuntu/mmdetection/data/train/claremont_2020_31.tif")






h, w = img.shape[:2]

aug = Resize((512, 512))

transform = aug.get_transform(img)
img_resized = transform.apply_image(img)
print(img_resized.shape, h, w)
img_tensor = img_resized.astype("float32").transpose(2, 0, 1)#[::-1]
img_tensor = torch.as_tensor(img_tensor.copy()).to(cfg.model.device)

# 4) Inference
with torch.no_grad():
    outputs = model([{"image": img_tensor, "height": h, "width": w}])
instances = outputs[0]["instances"].to("cpu")

# 1) score threshold
keep = instances.scores > 0.1
instances = instances[keep]
# 2) NMS
keep2 = torchvision.ops.nms(
    instances.pred_boxes.tensor,
    instances.scores,
    0.1
)
instances = instances[keep2]

        
# Prepare a Visualizer that won’t draw text
vis = Visualizer(
    img[:, :, ::-1],           # BGR→RGB for nicer colors
    metadata=None,             # no metadata → no class colors/names
    scale=1.0,
    instance_mode=ColorMode.IMAGE   # plain image mode
)

# Overlay boxes (and masks/keypoints if present), but no labels:
out = vis.overlay_instances(
    boxes=instances.pred_boxes if instances.has("pred_boxes") else None,
    masks=instances.pred_masks if instances.has("pred_masks") else None,
    keypoints=instances.pred_keypoints if instances.has("pred_keypoints") else None,
    labels=None                # <-- this suppresses all text
)

# Save out
cv2.imwrite("output_no_text.jpg", out.get_image()[:, :, ::-1])  # RGB→BGR back
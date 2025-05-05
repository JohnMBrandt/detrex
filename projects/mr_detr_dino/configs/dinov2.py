from detrex.config import get_config
from .models.mrdetr_dinov2 import model
from detectron2.config import LazyCall as L
import albumentations as A
from detectron2.data import transforms as T

# 1) grab the default dataloader
dataloader = get_config("common/data/custom_coco.py").dataloader



# get default config
#dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train


# modify training config
#train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth"
train.output_dir = "./output/dino_vithuge-512-augment-2048-mrdetr"
#train.init_weight_prefix = "backbone.net."

# max training iterations
train.max_iter = 120000

# run evaluation every 5000 iters
train.eval_period = 2000

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 2000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2
train.amp.enabled=True


# set training devices
train.device = "cuda"
model.device = train.device
#model.backbone.net.freeze_at = 6


# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
#optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 10

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 3

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

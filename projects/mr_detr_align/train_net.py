#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
#from detrex.projects.dino.modeling.dino import DINO
#from detectron2.structures.image_list import ImageList
from detectron2.utils.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128,garbage_collection_threshold:0.9"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("detrex")
import torchvision
from detectron2.structures import Instances
from torch import nn

class NMSWrapper(nn.Module):
    """
    Wrap any DETR-style model so that after forward it applies
    score-thresholding + torchvision.ops.nms to the Instances.
    """
    def __init__(self, model, score_thresh: float, nms_thresh: float):
        super().__init__()
        self.model = model
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh

    def forward(self, batched_inputs):
        # batched_inputs is a list[dict] or ImageList, but we only care about .tensor
        # so let the original model deal with it:
        outputs = self.model(batched_inputs)
        # DETR/DINO always returns a list[dict] with key "instances"
        new_outputs = []
        for out in outputs:
            inst: Instances = out["instances"].to("cpu")
            # 1) score threshold
            keep = inst.scores > self.score_thresh
            inst = inst[keep]
            # 2) NMS
            keep2 = torchvision.ops.nms(
                inst.pred_boxes.tensor,
                inst.scores,
                self.nms_thresh,
            )
            inst = inst[keep2]
            new_outputs.append({"instances": inst})
        return new_outputs


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


class Trainer(SimpleTrainer):
    """
    A SimpleTrainer + AMP + gradient accumulation.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
        accum_iters: int = 4,           
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        self.amp = amp
        if amp and grad_scaler is None:
            from torch.cuda.amp import GradScaler
            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

        self.clip_grad_params = clip_grad_params
        self.accum_iters = accum_iters
        self._accum_step = 0
        self._update_step = 0

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        data = next(self._data_loader_iter)
        device = next(self.model.parameters()).device
        for sample in data:
            if "instances" in sample:
                sample["instances"] = sample["instances"].to(device)

        # 1) forward + get loss dict
        loss_dict = self.model(data)
        losses = (
            loss_dict
            if isinstance(loss_dict, torch.Tensor)
            else sum(loss_dict.values())
        )

        # 2) scale the loss down by accum_iters
        loss_scaled = losses / self.accum_iters

        # 3) zero grad on first sub-step
        if self._accum_step == 0:
            self.optimizer.zero_grad()

        # 4) backward
        if self.amp:
            self.grad_scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # 5) on the last sub-step do optimizer step + grad clip + zero_grad
        self._accum_step += 1
        if self._accum_step == self.accum_iters:
            if self.amp:
                # unscale for grad clip
                if self.clip_grad_params is not None:
                    self.grad_scaler.unscale_(self.optimizer)
                    self._clip_grads()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                if self.clip_grad_params is not None:
                    self._clip_grads()
                self.optimizer.step()
            self._update_step += 1
            if self._update_step % 100 == 0:
                print("emptying the cache")
                torch.cuda.empty_cache()
            self._accum_step = 0

        # 6) log metrics every micro-step (or you can only log on step==accum_iters-1)
        self._write_metrics(loss_dict, 0.0)

    def _clip_grads(self):
        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, **self.clip_grad_params)

def do_test(cfg, model):
    # wrap the model so that every inference pass does NMS
    nms_model = NMSWrapper(
        model,
        score_thresh= 0.0,
        nms_thresh  = 0.1,
    )
    # run the normal detectron2 evaluation
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            nms_model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    import collections

    # 1) Build the model
    model = instantiate(cfg.model)

    # ─── DEBUG #1: Backbone parameter counts ───
    total_backbone = sum(p.numel() for p in model.backbone.vit.parameters())
    trainable_backbone = sum(p.numel() for p in model.backbone.vit.parameters() if p.requires_grad)
    print(f"[DEBUG] Backbone total params={total_backbone:,}, trainable={trainable_backbone:,}")
    # Show first few frozen vs unfrozen names
    frozen = []
    unfrozen = []
    for n, p in model.named_parameters():
        if n.startswith("backbone.vit"):
            (unfrozen if p.requires_grad else frozen).append(n)
    print("[DEBUG] Frozen backbone examples:", frozen[:25])
    print("[DEBUG] Unfrozen backbone examples:", unfrozen[:25])

    # 2) Move to device
    model.to(cfg.train.device)

    # 3) Build optimizer with two groups
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
            ],
            "lr": 1e-4,
            #"initial_lr": 1e-7,
        },
    ]
    optim = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)

    for i, g in enumerate(optim.param_groups):
        params = g["params"]
        # pick out names by identity check
        names_in_group = [
            n
            for n, p in model.named_parameters()
            if any(p is q for q in params)
        ]
        lr = g.get("lr", None)
        wd = g.get("weight_decay", g.get("weight_decay", None))
        print(f"[DEBUG] OPTIM GROUP {i}: lr={lr}, wd={wd}, num_params={len(names_in_group)}")
        print("          sample params:", names_in_group[:25])

    # 4) Build data loader & wrap for DDP
    train_loader = instantiate(cfg.dataloader.train)
    model = create_ddp_model(model, **cfg.train.ddp)

    # 5) Create trainer
    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
        accum_iters=getattr(cfg.train, "accum_iters", 4),
    )

    # 6) Checkpointer
    checkpointer = DetectionCheckpointer(model, cfg.train.output_dir, trainer=trainer)

    # 7) Hooks
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    # 8) Resume or load
    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    start_iter = trainer.iter + 1 if args.resume and checkpointer.has_checkpoint() else 0
    #start_iter = 4000
    # 9) Train
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

'''
param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not match_name_keywords(n, ["reference_points", "sampling_offsets"])
                and p.requires_grad
            ],
            "lr": 1e-4,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, ["reference_points", "sampling_offsets"])
                and p.requires_grad
            ],
            "lr": 1e-5,
        },
    ]
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, ["backbone"]) and p.requires_grad
            ],
            "lr": 1e-5,
        },
'''
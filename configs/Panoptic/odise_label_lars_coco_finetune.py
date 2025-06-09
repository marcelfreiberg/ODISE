from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler
import detectron2.data.transforms as T

from ..common.models.odise_with_label_lars_coco import model
from ..common.data.lars_coco_panoptic_semseg import dataloader
from ..common.train import train
from ..common.optim import AdamW as optimizer


# ---------------- Memory-safe augmentations ----------------
# Replace original augmentations with 768x768 crop (was 1024x1024)
dataloader.train.mapper.augmentations = [
    T.ResizeShortestEdge(
        short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768),
        max_size=1333,
        sample_style="choice",
    ),
    T.RandomFlip(horizontal=0.5),
    T.FixedSizeCrop(crop_size=(768, 768)),  # Reduced from 1024x1024
]

# ---------------- Memory-safe criterion ----------------
# Reduce point sampling from 12544 to 4096 points per image
model.criterion.num_points = 4096  # was 12544

# ------------------- Training schedule -------------------
# The schedule below equals ~50 epochs with batch-size 64.
# Feel free to reduce "train.max_iter" if your dataset is smaller.
train.max_iter = 40128            # 50 epochs * (dataset_size/ batch_size). Adjust as needed.
train.grad_clip = 0.01
train.checkpointer.period = 2000  # save every 2k iters
train.eval_period        = 2000

# Learning-rate scheduler (same shape as the COCO schedule but shorter)
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[int(train.max_iter * 0.9), int(train.max_iter * 0.95)],
        num_updates=train.max_iter,
    ),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.067,
)

# Optimizer
optimizer.lr = 5e-5   # smaller lr for fine-tuning
optimizer.weight_decay = 0.05 

dataloader.train.total_batch_size = 32
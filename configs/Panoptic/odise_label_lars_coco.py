from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler
import detectron2.data.transforms as T

from ..common.models.odise_with_label_lars_coco import model
from ..common.data.lars_coco_panoptic_semseg import dataloader
from ..common.train import train
from ..common.optim import AdamW as optimizer


dataloader.train.mapper.augmentations = [
    T.ResizeShortestEdge(
        short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768),
        max_size=1333,
        sample_style="choice",
    ),
    T.RandomFlip(horizontal=0.5),
    T.FixedSizeCrop(crop_size=(768, 768)),  # Reduced from 1024x1024
]

dataloader.test.local_batch_size = 1

# Fewer sampled points → lower VRAM
model.criterion.num_points = 8192  # was 12544

dataloader.train.total_batch_size = 8

train.max_iter       = 6_000
train.grad_clip      = 0.01
train.checkpointer.period = 500

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[int(train.max_iter * 0.9), int(train.max_iter * 0.95)],  # 3663, 3866
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,  # 250 iterations (≈ 0.75 epoch)
    warmup_factor=0.067,
)

train.auto_scale_lr = dict(enabled=False)

optimizer.lr = 1e-4
optimizer.weight_decay = 0.05

# Override prompts for water domain shift testing
# model.category_head.prompt = "maritime"
# model.clip_head.prompt = "maritime"

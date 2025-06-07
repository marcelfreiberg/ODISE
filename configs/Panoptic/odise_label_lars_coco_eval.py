# This config is for evaluation only on LARS COCO dataset
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ..common.models.odise_with_label_lars_coco import model
from ..common.data.lars_coco_panoptic_semseg import dataloader
from ..common.train import train
from ..common.optim import AdamW as optimizer

# Training configs (not used for eval-only but kept for compatibility)
train.max_iter = 92_188
train.grad_clip = 0.01
train.checkpointer.period = 4500

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[163889, 177546],
        num_updates=184375,
    ),
    warmup_length=500 / 184375,
    warmup_factor=0.067,
)

optimizer.lr = 1e-4
optimizer.weight_decay = 0.05 
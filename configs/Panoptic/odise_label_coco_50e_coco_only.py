# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ODISE/blob/main/LICENSE
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ..common.models.odise_with_label import model
from ..common.data.coco_panoptic_semseg import dataloader
from ..common.train import train
from ..common.optim import AdamW as optimizer

train.max_iter = 92_188
train.grad_clip = 0.01
train.checkpointer.period = 4500

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        # assume 100e with batch-size 64 as original LSJ
        # Equivalent to 100 epochs.
        # 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
        milestones=[163889, 177546],
        num_updates=184375,
    ),
    # for warmup length we adopted COCO LSJ setting
    warmup_length=500 / 184375,
    warmup_factor=0.067,
)

optimizer.lr = 1e-4
optimizer.weight_decay = 0.05

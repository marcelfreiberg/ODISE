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

# ---------------- Criterion ----------------
# Fewer sampled points → lower VRAM
model.criterion.num_points = 4096  # was 12544

# ---------------- Data loader ----------------
dataloader.train.total_batch_size = 32

# ---------------- Training schedule ----------------
# 2606 images / 32  ⇒  82 iters per epoch  ⇒  50 epochs → 4 100 iters
train.max_iter       = 4100
train.grad_clip      = 0.01
train.checkpointer.period = 500            # save ~every 6 epochs
train.eval_period    = 82                  # validate every epoch

# Learning-rate scheduler (milestones adjusted for new max_iter)
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[int(train.max_iter * 0.9), int(train.max_iter * 0.95)],  # 3663, 3866
        num_updates=train.max_iter,
    ),
    warmup_length=1000 / train.max_iter,  # 1 k-iter warm-up (≈ 12 epochs)
    warmup_factor=0.067,
)

# ---------------- Optimiser ----------------
optimizer.lr            = 1e-4             # up from 5 e-5
optimizer.weight_decay  = 0.05             # keep default for ViT-L

# ----------------------------------------------------------
# TEST / VALIDATION MAPPER – keep memory identical to training
# ----------------------------------------------------------
# remove the FixedSizeCrop for test-time (centre crop hurts metrics)  
# dataloader.test.mapper.augmentations = [
#     T.ResizeShortestEdge(
#         short_edge_length=768,
#         max_size=1333,
#         sample_style="choice",
#     )
# ]

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import DatasetMapper

from odise.data import (
    COCOPanopticDatasetMapper,
    build_d2_test_dataloader,
    build_d2_train_dataloader,
    get_openseg_labels,
)
from odise.evaluation.d2_evaluator import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    SemSegEvaluator,
)
from odise.modeling.wrapper.pano_wrapper import OpenPanopticInference
from detectron2.data import MetadataCatalog

dataloader = OmegaConf.create()

dataloader.train = L(build_d2_train_dataloader)(
    dataset=L(get_detection_dataset_dicts)(
        names="lars_coco_train_panoptic_with_sem_seg", filter_empty=True
    ),
    mapper=L(COCOPanopticDatasetMapper)(
        is_train=True,
        # COCO LSJ aug
        augmentations=[
            L(T.RandomFlip)(horizontal=True),
            L(T.ResizeScale)(
                min_scale=0.1,
                max_scale=2.0,
                target_height=1024,
                target_width=1024,
            ),
            L(T.FixedSizeCrop)(crop_size=(1024, 1024)),
        ],
        image_format="RGB",
    ),
    total_batch_size=64,
    num_workers=4,
)

dataloader.test = L(build_d2_test_dataloader)(
    dataset=L(get_detection_dataset_dicts)(
        names="lars_coco_val_panoptic_with_sem_seg",
        filter_empty=False,
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=1024, sample_style="choice", max_size=2560),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    local_batch_size=1,
    num_workers=1,
)

dataloader.evaluator = [
    # L(COCOEvaluator)(
    #     dataset_name="${...test.dataset.names}",
    #     tasks=("segm",),
    # ),
    L(SemSegEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
    L(COCOPanopticEvaluator)(
        dataset_name="${...test.dataset.names}",
    ),
]

dataloader.wrapper = L(OpenPanopticInference)(
    labels=L(get_openseg_labels)(dataset="lars_coco", prompt_engineered=True),
    metadata=L(MetadataCatalog.get)(name="${...test.dataset.names}"),
)
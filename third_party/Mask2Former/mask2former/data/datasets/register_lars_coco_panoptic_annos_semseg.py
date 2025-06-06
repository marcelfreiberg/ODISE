# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


# Lars COCO specific categories - these IDs are not continuous
LARS_COCO_CATEGORIES = [
    {"id": 1, "name": "Static Obstacle", "isthing": 0, "color": [220, 20, 60]},
    {"id": 3, "name": "Water", "isthing": 0, "color": [119, 11, 32]},
    {"id": 5, "name": "Sky", "isthing": 0, "color": [0, 0, 142]},
    {"id": 11, "name": "Boat/ship", "isthing": 1, "color": [0, 0, 230]},
    {"id": 12, "name": "Row boats", "isthing": 1, "color": [106, 0, 228]},
    {"id": 13, "name": "Paddle board", "isthing": 1, "color": [0, 60, 100]},
    {"id": 14, "name": "Buoy", "isthing": 1, "color": [0, 80, 100]},
    {"id": 15, "name": "Swimmer", "isthing": 1, "color": [0, 0, 70]},
    {"id": 16, "name": "Animal", "isthing": 1, "color": [0, 0, 192]},
    {"id": 17, "name": "Float", "isthing": 1, "color": [250, 170, 30]},
    {"id": 19, "name": "Other", "isthing": 1, "color": [100, 170, 30]},
]

_PREDEFINED_SPLITS_LARS_COCO_PANOPTIC = {
    "lars_coco_train_panoptic": (
        "panoptic_train",
        "annotations/panoptic_train.json",  
        "panoptic_semseg_train",
    ),
    "lars_coco_val_panoptic": (
        "panoptic_val",
        "annotations/panoptic_val.json",  
        "panoptic_semseg_val",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in LARS_COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in LARS_COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in LARS_COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in LARS_COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(LARS_COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def load_lars_coco_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/lars_coco/images/train".
        gt_dir (str): path to the raw annotations. e.g., "~/lars_coco/panoptic_train".
        json_file (str): path to the json file. e.g., "~/lars_coco/annotations/panoptic_train.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # Lars COCO uses .jpg extension
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_lars_coco_panoptic_annos_sem_seg(
    name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json
):
    panoptic_name = name
    if panoptic_name in MetadataCatalog:
        if hasattr(MetadataCatalog.get(panoptic_name), "thing_classes"):
            delattr(MetadataCatalog.get(panoptic_name), "thing_classes")
        if hasattr(MetadataCatalog.get(panoptic_name), "thing_colors"):
            delattr(MetadataCatalog.get(panoptic_name), "thing_colors")
    MetadataCatalog.get(panoptic_name).set(
        thing_classes=metadata["thing_classes"],
        thing_colors=metadata["thing_colors"],
    )

    # the name is "lars_coco_train_panoptic_with_sem_seg" and "lars_coco_val_panoptic_with_sem_seg"
    semantic_name = name + "_with_sem_seg"
    DatasetCatalog.register(
        semantic_name,
        lambda: load_lars_coco_panoptic_json(panoptic_json, image_root, panoptic_root, sem_seg_root, metadata),
    )
    MetadataCatalog.get(semantic_name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_lars_coco_panoptic_annos_sem_seg(root):
    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_LARS_COCO_PANOPTIC.items():
        
        if 'train' in prefix:
            image_root = os.path.join(root, "images/train")
        elif 'val' in prefix:
            image_root = os.path.join(root, "images/val")

        # For LARS COCO, we don't have instance annotations, so we use None
        instances_json = None

        register_lars_coco_panoptic_annos_sem_seg(
            prefix,
            get_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )


_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "lars_coco"
register_all_lars_coco_panoptic_annos_sem_seg(_root) 
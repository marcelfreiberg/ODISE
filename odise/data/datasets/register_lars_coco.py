import json
import os
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
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
    
    # Sort categories by original ID to ensure consistent ordering (same as preparation script)
    sorted_categories = sorted(LARS_COCO_CATEGORIES, key=lambda x: x["id"])
    
    # Split thing vs. stuff classes
    thing_classes = [k["name"] for k in sorted_categories if k["isthing"] == 1]
    thing_colors = [k["color"] for k in sorted_categories if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in sorted_categories if k["isthing"] == 0]
    stuff_colors = [k["color"] for k in sorted_categories if k["isthing"] == 0]
    
    # For semantic evaluation, we need all classes in a unified list
    # Following COCO pattern: all classes get contiguous IDs [0, N-1]
    all_classes = [k["name"] for k in sorted_categories]
    all_colors = [k["color"] for k in sorted_categories]
    
    # Create unified contiguous ID mapping (like COCO does)
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    contiguous_id_to_class_name = []
    
    # Assign contiguous IDs in sorted order
    for i, cat in enumerate(sorted_categories):
        if cat["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # For semantic evaluation, ALL classes go in stuff_dataset_id_to_contiguous_id
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i
        contiguous_id_to_class_name.append(cat["name"])
    
    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = all_classes  # All classes for semantic evaluation
    meta["stuff_colors"] = all_colors    # All colors for semantic evaluation 
    meta["sem_stuff_classes"] = all_classes  # For text embeddings
    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
    meta["contiguous_id_to_class_name"] = contiguous_id_to_class_name
    meta["dataname"] = "lars_coco_val_panoptic"

    return meta


def load_lars_coco_panoptic_json(root, json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to "~/<root>/images/train" or "…/images/val"
        gt_dir (str): path to "~/<root>/<panoptic_root>"
        json_file (str): path to "~/<root>/annotations/panoptic_*.json"
        semseg_dir (str): path to "~/<root>/<panoptic_semseg_*>"
    Returns:
        list[dict]: Detectron2-format dicts
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
        # images end in .jpg
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        pan_seg_file = os.path.join(gt_dir, ann["file_name"])
        sem_seg_file = os.path.join(semseg_dir, ann["file_name"])

        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": pan_seg_file,
                "sem_seg_file_name": sem_seg_file,
                "segments_info": segments_info,
                "meta": {"dataname": meta["dataname"]},  # Required by model
            }
        )

    assert len(ret), f"No images found in {image_dir}!"
    first = ret[0]
    assert PathManager.isfile(first["file_name"]), first["file_name"]
    assert PathManager.isfile(first["pan_seg_file_name"]), first["pan_seg_file_name"]
    assert PathManager.isfile(first["sem_seg_file_name"]), first["sem_seg_file_name"]
    return ret


def register_lars_coco_panoptic_annos_sem_seg(
    root, name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root
):
    DatasetCatalog.register(
        name,
        lambda: load_lars_coco_panoptic_json(
            root, panoptic_json, image_root, panoptic_root, sem_seg_root, metadata
        ),
    )
    MetadataCatalog.get(name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        dataset_name=name,
        **metadata,
    )


def register_all_lars_coco_panoptic_annos_sem_seg(root):
    for prefix, (panoptic_root, panoptic_json, semantic_root) in _PREDEFINED_SPLITS_LARS_COCO_PANOPTIC.items():
        if "train" in prefix:
            image_root = os.path.join(root, "images/train")
        elif "val" in prefix:
            image_root = os.path.join(root, "images/val")
        else:
            raise ValueError(f"Unknown split prefix: {prefix}")

        register_lars_coco_panoptic_annos_sem_seg(
            root,
            prefix,
            get_metadata(),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
        )


def load_lars_coco_semantic_only(image_dir, sem_seg_dir, meta):
    """Load semantic segmentation dataset with required meta key"""
    # Use built-in loader and add meta key
    dataset_dicts = load_sem_seg(sem_seg_dir, image_dir, gt_ext="png", image_ext="jpg")
    
    # Add required meta key to each item
    for item in dataset_dicts:
        item["meta"] = {"dataname": meta["dataname"]}
    
    return dataset_dicts


# Also register semantic-only datasets for pure semantic evaluation
def register_lars_coco_semantic_only():
    """Register LARS COCO for semantic segmentation evaluation only"""
    root = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "lars_coco"
    meta = get_metadata()
    
    for split in ["train", "val"]:
        image_dir = os.path.join(root, f"images/{split}")
        sem_seg_dir = os.path.join(root, f"panoptic_semseg_{split}")
        dataset_name = f"lars_coco_{split}_sem_seg"
        
        # Update metadata dataname for this specific dataset
        split_meta = meta.copy()
        split_meta["dataname"] = dataset_name
        
        # Register semantic segmentation dataset with our custom loader
        DatasetCatalog.register(
            dataset_name,
            lambda x=image_dir, y=sem_seg_dir, m=split_meta: load_lars_coco_semantic_only(x, y, m),
        )
        MetadataCatalog.get(dataset_name).set(
            image_root=image_dir,
            sem_seg_root=sem_seg_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            dataset_name=dataset_name,
            **split_meta,
        )


_root = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "lars_coco"
register_all_lars_coco_panoptic_annos_sem_seg(_root)
register_lars_coco_semantic_only() 
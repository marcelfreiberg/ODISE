#!/usr/bin/env python3
# Generate semantic segmentation masks from LARS COCO panoptic annotations
# This includes ALL categories (both stuff and things) in the semantic masks
# which allows proper semantic evaluation of maritime objects

import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from panopticapi.utils import rgb2id
from PIL import Image
from pathlib import Path


# LARS COCO categories - same as in registration
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


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    """Convert panoptic segmentation to semantic segmentation"""
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255  # Initialize with ignore label
    
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    
    Image.fromarray(output).save(output_semantic)


def separate_lars_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations for LARS COCO dataset.
    
    Unlike the original LARS preparation, this maps ALL categories (both stuff and things)
    to contiguous semantic IDs, enabling proper semantic evaluation of maritime objects.
    
    Args:
        panoptic_json (str): path to the panoptic json file
        panoptic_root (str): directory with panoptic annotation files
        sem_seg_root (str): directory to output semantic annotation files
        categories (list[dict]): category metadata
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    # Create mapping from original category ID to contiguous semantic ID
    # Sort by original ID to ensure consistent ordering
    sorted_categories = sorted(categories, key=lambda x: x["id"])
    id_map = {}
    
    print("Category mapping (original_id -> semantic_id):")
    for i, cat in enumerate(sorted_categories):
        id_map[cat["id"]] = i
        print(f"  {cat['id']:2} ({cat['name']:15}) -> {i}")
    
    print(f"ID mapping: {id_map}")
    assert len(categories) <= 254, "Too many categories for semantic segmentation"

    with open(panoptic_json) as f:
        obj = json.load(f)

    # Use multiprocessing for faster conversion
    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input_file = os.path.join(panoptic_root, file_name)
            output_file = os.path.join(sem_seg_root, file_name)
            yield input_file, output_file, segments

    print(f"Start writing semantic masks to {sem_seg_root} ...")
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=50,
    )
    pool.close()
    pool.join()
    print(f"Finished. time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "datasets")) / "lars_coco"
    
    print("Generating LARS COCO semantic segmentation masks from panoptic annotations...")
    print("This includes ALL categories (stuff + things) for proper semantic evaluation.")
    
    for split in ["train", "val"]:
        print(f"\nProcessing {split} split...")
        
        panoptic_json = dataset_dir / f"annotations/panoptic_{split}.json"
        panoptic_root = dataset_dir / f"panoptic_{split}"
        
        # Create new semantic directory with suffix to avoid overwriting
        sem_seg_root = dataset_dir / f"panoptic_semseg_{split}_full"
        
        if not panoptic_json.exists():
            print(f"Warning: {panoptic_json} not found, skipping {split}")
            continue
            
        separate_lars_coco_semantic_from_panoptic(
            str(panoptic_json),
            str(panoptic_root),
            str(sem_seg_root),
            LARS_COCO_CATEGORIES,
        )
    
    print("\nDone! Generated semantic masks include all categories:")
    print("- Static Obstacle, Water, Sky (stuff categories)")  
    print("- Boat/ship, Row boats, Buoy, etc. (thing categories)")
    print("\nTo use these for evaluation, update your dataset registration to point to")
    print("the new semantic directories: panoptic_semseg_*_full") 
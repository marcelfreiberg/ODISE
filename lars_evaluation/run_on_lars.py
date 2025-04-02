import os
import subprocess
import platform
import argparse
from datetime import datetime
from contextlib import ExitStack
import torch
from PIL import Image
import numpy as np
import tqdm
import time
import json

from detectron2.config import instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.evaluation import inference_context
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color

from odise import model_zoo
from odise.checkpoint import ODISECheckpointer
from odise.config import instantiate_odise
from odise.modeling.wrapper import OpenPanopticInference

setup_logger()
logger = setup_logger(name="odise")

LARS_STUFF_CLASSES = [
    ["Static Obstacle"],
    ["Water"],
    ["Sky"],
]

LARS_STUFF_CLASSES_EXTENDED = [
    [
        "Static Obstacle",
        "Fixed Object",
        "Immovable Object",
        "Barrier",
        "Structure",
        "Terrain",
        "Ground",
        "Land",
        "Forest",
        "Trees",
        "Vegetation",
        "Plants",
        "Grass",
        "Bush",
        "Foliage",
        "Beach",
        "Shore",
        "Coast",
        "Sand",
        "Rocks",
        "Cliff",
        "Stone",
        "Boulder",
        "Background",
        "Environment",
        "Surroundings",
        "Landscape",
        "Dock",
        "Pier",
        "Jetty",
        "Building",
        "Construction",
    ],
    [
        "Water",
        "Sea",
        "Ocean",
        "Lake",
        "River",
        "Pond",
        "Fluid",
        "Aquatic surface",
        "Waterway",
        "Stream",
        "Waves",
        "Liquid",
    ],
    [
        "Sky",
        "Clouds",
        "Atmosphere",
        "Heavens",
        "Air",
        "Horizon",
        "Celestial",
        "Firmament",
    ],
]

LARS_STUFF_COLORS = [
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164],
]

LARS_THING_CLASSES = [
    ["Boat/ship"],
    ["Row boats"],
    ["Paddle board"],
    ["Buoy"],
    ["Swimmer"],
    ["Animal"],
    ["Float"],
    ["Other"],
]

LARS_THING_COLORS = [
    [255, 87, 51],  # Boat/ship - bright orange-red
    [50, 205, 50],  # Row boats - lime green
    [255, 0, 255],  # Paddle board - magenta
    [255, 215, 0],  # Buoy - gold
    [0, 128, 128],  # Swimmer - teal
    [139, 69, 19],  # Animal - saddle brown
    [220, 20, 60],  # Float - crimson
    [169, 169, 169],  # Other - dark gray
]

ODISE_TO_LARS_MAPPING = {
    0: 11,
    1: 12,
    2: 13,
    3: 14,
    4: 15,
    5: 16,
    6: 17,
    7: 19,
    8: 1,
    9: 3,
    10: 5,
}


def build_demo_classes_and_metadata(vocab, label_list):
    extra_classes = []

    if vocab:
        for words in vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])
    extra_colors = [
        random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))
    ]

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if "LARS" in label_list:
        demo_thing_classes += LARS_THING_CLASSES
        demo_stuff_classes += LARS_STUFF_CLASSES
        demo_thing_colors += LARS_THING_COLORS
        demo_stuff_colors += LARS_STUFF_COLORS
    if "LARS_EXTENDED" in label_list:
        demo_thing_classes += LARS_THING_CLASSES
        demo_stuff_classes += LARS_STUFF_CLASSES_EXTENDED
        demo_thing_colors += LARS_THING_COLORS
        demo_stuff_colors += LARS_STUFF_COLORS

    MetadataCatalog.pop("odise_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }

    demo_classes = demo_thing_classes + demo_stuff_classes

    return demo_classes, demo_metadata


def get_gpu_info():
    """Get GPU information using nvidia-smi command."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip().split("\n")
    except Exception as e:
        return [f"Error getting GPU info: {e}"]


def get_system_info():
    """Get system information."""
    system_info = {
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "Number of GPUs": torch.cuda.device_count(),
        "GPUs": get_gpu_info(),
    }

    print("--------------------------------")
    print("System Information:")
    for key, value in system_info.items():
        if key == "GPUs":
            print(f"  {key}:")
            for i, gpu in enumerate(value):
                print(f"    GPU {i}: {gpu}")
        else:
            print(f"  {key}: {value}")
    print("--------------------------------")

    return system_info


class VisualizationDemo(object):
    def __init__(self, model, metadata, aug, instance_mode=ColorMode.IMAGE):
        """
        Args:
            model (nn.Module):
            metadata (MetadataCatalog): image metadata.
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def predict(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions

    def run_on_image(self, image, predictions):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        # predictions = self.predict(image)
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        panoptic_seg, segments_info = predictions["panoptic_seg"]
        vis_output = visualizer.draw_panoptic_seg(
            panoptic_seg.to(self.cpu_device), segments_info
        )

        return vis_output

    def run_on_image_lars(self, image, predictions):
        # predictions = self.predict(image)

        panoptic_seg, segments_info = predictions["panoptic_seg"]

        if isinstance(panoptic_seg, torch.Tensor):
            panoptic_seg = panoptic_seg.cpu().numpy()

        panoptic_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        for segment in segments_info:
            segment_id = segment["id"]
            is_thing = segment["isthing"]
            category_id = ODISE_TO_LARS_MAPPING[segment["category_id"]]

            binary_mask = panoptic_seg == segment_id

            instance_id = segment_id if is_thing else 0

            g_channel = instance_id // 256
            b_channel = instance_id % 256

            panoptic_mask[binary_mask, 0] = category_id
            panoptic_mask[binary_mask, 1] = g_channel
            panoptic_mask[binary_mask, 2] = b_channel

        return panoptic_mask


def inference(image, model, vocab, label_list, aug, lars_format=False):

    image = np.array(image)

    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
    with ExitStack() as stack:
        inference_model = OpenPanopticInference(
            model=model,
            labels=demo_classes,
            metadata=demo_metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True,
        )
        stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())

        demo = VisualizationDemo(inference_model, demo_metadata, aug)

        # Measure inference time
        start_time = time.time()
        predictions = demo.predict(image)
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time

        if not lars_format:
            out = demo.run_on_image(image, predictions).get_image()
            return Image.fromarray(out), fps
        else:
            out = demo.run_on_image(image, predictions).get_image()
            out_lars = demo.run_on_image_lars(image, predictions)
            return Image.fromarray(out), Image.fromarray(out_lars), fps


def main():
    parser = argparse.ArgumentParser(description="Run ODISE on LaRS dataset")
    parser.add_argument(
        "--init-from",
        type=str,
        help="Init from the given checkpoint",
        default="/data/mfreiberg/weights/ODISE/odise_label_coco_50e-b67d2efc.pth",
    )
    parser.add_argument(
        "--lars-dir",
        type=str,
        default="/data/mfreiberg/datasets/lars",
        help="Path to LaRS dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/mfreiberg/predictions",
        help="Base directory to save output",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name for the output directory (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--label-list",
        type=str,
        default="LARS",
        choices=["LARS", "LARS_EXTENDED"],
        help="Label list to use",
    )
    args = parser.parse_args()

    system_info = get_system_info()

    cfg = model_zoo.get_config("Panoptic/odise_label_coco_50e.py", trained=True)

    cfg.model.overlap_threshold = 0
    seed_all_rng(42)

    dataset_cfg = cfg.dataloader.test
    wrapper_cfg = cfg.dataloader.wrapper

    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(args.init_from)

    vocab = ""
    label_list = [args.label_list]

    out_dir = os.path.join(
        args.output_dir, f"odise_{args.label_list.lower()}_{args.split}"
    )
    out_dir_panop = os.path.join(out_dir, "panop")
    out_dir_lars_format = os.path.join(out_dir, "lars_format")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_panop, exist_ok=True)
    os.makedirs(out_dir_lars_format, exist_ok=True)

    # Save system info to a JSON file
    with open(os.path.join(out_dir, "system_info.json"), "w") as f:
        json.dump(system_info, f, indent=4)

    lars_val_dir = os.path.join(args.lars_dir, args.split)
    image_list_path = os.path.join(lars_val_dir, "image_list.txt")

    with open(image_list_path, "r") as f:
        image_names = [line.strip() for line in f.readlines()]

    fps_values = []
    for image_name in tqdm.tqdm(image_names, desc="Processing images"):
        img_path = os.path.join(lars_val_dir, "images", f"{image_name}.jpg")
        image = Image.open(img_path)

        (out_img, out_img_lars_format, fps) = inference(
            image, model, vocab, label_list, aug, lars_format=True
        )
        fps_values.append(fps)
        out_img.save(os.path.join(out_dir_panop, f"{image_name}.png"))
        out_img_lars_format.save(os.path.join(out_dir_lars_format, f"{image_name}.png"))

    # Calculate and save FPS statistics
    fps_stats = {
        "mean_fps": np.mean(fps_values),
        "median_fps": np.median(fps_values),
        "min_fps": np.min(fps_values),
        "max_fps": np.max(fps_values),
        "std_fps": np.std(fps_values),
        "fps_per_image": dict(zip([f"{name}.jpg" for name in image_names], fps_values)),
    }

    # Save FPS statistics to a JSON file
    with open(os.path.join(out_dir, "fps_stats.json"), "w") as f:
        json.dump(fps_stats, f, indent=4)

    print(f"Average FPS: {fps_stats['mean_fps']:.2f}")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()

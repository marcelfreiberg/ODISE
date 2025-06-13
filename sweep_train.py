"""
W&B Sweep wrapper script for ODISE training.
This script handles parameter overrides from W&B sweeps and calls the original training script.
"""

import os
import sys
import subprocess
import wandb
import argparse


def build_config_overrides(config):
    """Convert W&B config parameters to DetectronLazy config overrides."""
    overrides = []
    
    # Learning rate parameters
    if 'lr' in config:
        overrides.append(f"optimizer.lr={config.lr}")
    
    if 'weight_decay' in config:
        overrides.append(f"optimizer.weight_decay={config.weight_decay}")
    
    # Training parameters
    if 'max_iter' in config:
        overrides.append(f"train.max_iter={config.max_iter}")
    
    if 'grad_clip' in config:
        overrides.append(f"train.grad_clip={config.grad_clip}")
    
    # Loss weights
    if 'dice_weight' in config:
        overrides.append(f"model.criterion.dice_weight={config.dice_weight}")
    
    if 'mask_weight' in config:
        overrides.append(f"model.criterion.mask_weight={config.mask_weight}")
    
    # Learning rate schedule parameters
    if 'warmup_length' in config and 'max_iter' in config:
        # Convert relative warmup length to absolute iterations
        warmup_iters = int(config.warmup_length * config.max_iter)
        warmup_ratio = warmup_iters / config.max_iter
        overrides.append(f"lr_multiplier.warmup_length={warmup_ratio}")
    
    if 'warmup_factor' in config:
        overrides.append(f"lr_multiplier.warmup_factor={config.warmup_factor}")
    
    # LR decay milestones
    if 'lr_decay_90' in config and 'lr_decay_95' in config and 'max_iter' in config:
        milestone_90 = int(config.max_iter * 0.9)
        milestone_95 = int(config.max_iter * 0.95)
        overrides.append(f"lr_multiplier.scheduler.values=[1.0,{config.lr_decay_90},{config.lr_decay_95}]")
        overrides.append(f"lr_multiplier.scheduler.milestones=[{milestone_90},{milestone_95}]")
        overrides.append(f"lr_multiplier.scheduler.num_updates={config.max_iter}")
    
    return overrides


def main():
    run = wandb.init(project="ODISE")
    config = run.config
    
    run_tag = f"sweep_{run.id}"
    
    cmd = [
        "python", "./tools/train_net.py",
        "--config-file", "./configs/Panoptic/odise_label_lars_coco.py",
        "--num-gpus", "8", 
        "--init-from", "/data/mfreiberg/weights/odise/odise_label_coco_50e-b67d2efc.pth",
        "--wandb",
        "--tag", run_tag
    ]
    
    cmd.extend(build_config_overrides(config))
    
    print("Running:", " ".join(cmd))
    print("Overrides:", build_config_overrides(config))
    print("W&B Run ID:", run.id)
    
    wandb.finish()
    
    completed = subprocess.run(cmd)
    return_code = completed.returncode
 
    run.finish(exit_code=return_code)
    return return_code


if __name__ == "__main__":
    sys.exit(main()) 
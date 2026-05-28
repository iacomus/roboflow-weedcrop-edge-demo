#!/usr/bin/env python
"""
Local RF-DETR Nano training on the weed-crop-aerial dataset (Apple Silicon / MPS).

Why local: Roboflow's hosted training is free, but exporting the trained model
to CoreML for on-device iOS deployment is a paid (Core-plan) feature. The
rf-detr package is open source (Apache-2.0), so we train + export CoreML
ourselves from our own dataset — fully sanctioned, $0.

Run (from training/, with the venv active):
    PYTORCH_ENABLE_MPS_FALLBACK=1 python train.py
"""
import os
import time

# Some DETR ops (e.g. bicubic upsample) aren't implemented on MPS yet;
# fall back to CPU for those rather than crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from rfdetr import RFDETRSmall

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")     # contains train/ valid/ test/ with _annotations.coco.json
OUTPUT_DIR = os.path.join(HERE, "output")

def main():
    print(f"[train] start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[train] dataset_dir = {DATASET_DIR}")
    print(f"[train] output_dir  = {OUTPUT_DIR}")

    model = RFDETRSmall()
    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        device="mps",            # Apple Silicon GPU
        epochs=50,               # early stopping will cut this short if it plateaus
        batch_size=4,            # safe for 16 GB unified memory
        grad_accum_steps=4,      # effective batch 16
        checkpoint_interval=5,   # save every 5 epochs -> always have a recent checkpoint
        early_stopping=True,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        num_workers=2,
        tensorboard=True,
    )
    print(f"[train] done {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

"""CRUX hold/volume instance segmentation training — D0.5 spike.

Structure follows the official RF-DETR segmentation fine-tuning cookbook:
https://github.com/roboflow/rf-detr/blob/main/docs/cookbooks/fine-tune_segmentation.ipynb
(Python API, NOT the CLI.)

License chain: rfdetr (Apache-2.0) + DINOv2 backbone + our data. Kaggle data is
research-only (PLAN §2 rule 2) — training with --dataset data/crux-dataset is a
research checkpoint and must NEVER ship in the app.

Usage:
    python tools/train_hold_seg.py                    # research spike, 384px, 2 classes
    python tools/train_hold_seg.py --fast-dev-run     # 1 train + 1 val batch, chain check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rfdetr import RFDETRSegSmall
from rfdetr.config import SegmentationTrainConfig
from rfdetr.datasets.aug_configs import AUG_CONSERVATIVE
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
from rfdetr.training.callbacks.best_model import BestModelCallback
from rfdetr.utilities.reproducibility import seed_all

SEED = 42
NUM_CLASSES = 2          # hold(0) + volume(1)
RESOLUTION = 384         # Gate C will benchmark 384/432/512; 384 for the spike
DATASET_DIR = Path("data/crux-dataset")
OUTPUT_DIR = Path("output/rfdetr_seg_small")   # overridable via --output-dir
FINAL_CKPT = OUTPUT_DIR / "crux-hold-seg-research.pth"
CLASS_NAMES = ["hold", "volume"]


def main(fast_dev_run: bool, batch_size: int, num_workers: int, resume: str | None,
         resolution: int = RESOLUTION, multi_scale: bool = True, epochs: int = 100,
         output_dir: str = "output/rfdetr_seg_small") -> None:
    global OUTPUT_DIR, FINAL_CKPT
    OUTPUT_DIR = Path(output_dir)
    FINAL_CKPT = OUTPUT_DIR / "crux-hold-seg-research.pth"
    seed_all(SEED)
    variant = RFDETRSegSmall(num_classes=NUM_CLASSES, resolution=resolution)
    variant.model_config.model_name = type(variant).__name__

    train_config = SegmentationTrainConfig(
        dataset_file="coco",                       # local COCO, Roboflow layout
        dataset_dir=str(DATASET_DIR),
        output_dir=str(OUTPUT_DIR),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=4 if not fast_dev_run else 1,  # effective batch 4 at bs=1
        num_workers=num_workers,                   # 0 on Windows: spawn workers crash
        resume=resume,
        multi_scale=multi_scale,                   # parameterized (on by default; author-match)
        aug_config=AUG_CONSERVATIVE,               # brightness/contrast jitter: lighting robustness
        use_ema=False,
        run_test=False,
        compute_train_metrics=True,
        compute_val_loss=True,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
        class_names=CLASS_NAMES,
        notes={
            "project": "CRUX hold segmentation spike",
            "classes": CLASS_NAMES,
            "data": "Kaggle bh set (research only, PLAN §2)",
            "model": "rf-detr-seg-small",
        },
        progress_bar="tqdm",
    )

    datamodule = RFDETRDataModule(variant.model_config, train_config)
    model = RFDETRModelModule(variant.model_config, train_config)
    trainer = build_trainer(train_config, variant.model_config)
    if fast_dev_run:
        trainer.fast_dev_run = True
    trainer.fit(model, datamodule=datamodule, ckpt_path=train_config.resume or None)

    if not fast_dev_run:
        raw = getattr(model.model, "_orig_mod", model.model)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            BestModelCallback._build_checkpoint_payload(
                raw.state_dict(),
                train_config.model_dump(),
                trainer,
                model_name="RFDETRSegSmall",
                model_config_dict=variant.model_config.model_dump(),
            ),
            FINAL_CKPT,
        )
        print(f"saved research checkpoint: {FINAL_CKPT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1)  # 8GB VRAM: bs=1 at 384px seg
    ap.add_argument("--num-workers", type=int, default=0)  # 0 on Windows (spawn worker crashes)
    ap.add_argument("--resume", type=str, default=None, help="path to .ckpt to resume training")
    ap.add_argument("--resolution", type=int, default=RESOLUTION, help="input size (multiple of 24)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (incremental short runs)")
    ap.add_argument("--output-dir", type=str, default="output/rfdetr_seg_small")
    ap.add_argument("--multi-scale", dest="multi_scale", action="store_true", default=True,
                    help="multi-scale training (default on; author-match)")
    ap.add_argument("--no-multi-scale", dest="multi_scale", action="store_false")
    args = ap.parse_args()
    epochs = args.epochs if args.epochs else (2 if args.fast_dev_run else 100)
    main(args.fast_dev_run, args.batch_size, args.num_workers, args.resume,
         resolution=args.resolution, multi_scale=args.multi_scale, epochs=epochs,
         output_dir=args.output_dir)

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import tools.program as program
from ppocr.utils.utility import set_seed

# импортируй main из твоего train.py
from tools.train import main as paddleocr_train_main


@contextmanager
def patched_argv(args: list[str]):
    old_argv = sys.argv[:]
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = old_argv


class PaddleOCRFineTuner:
    """
    YOLO-like wrapper для дообучения PaddleOCR моделей.

    Пример:
        model = PaddleOCRFineTuner("configs/rec/PP-OCRv6/rec.yml")
        model.train(
            pretrained_model="models/PP-OCRv6_medium_rec/best_accuracy",
            data_dir="dataset/ocr",
            train_labels="dataset/ocr/train.txt",
            val_labels="dataset/ocr/val.txt",
            epochs=50,
            batch_size=32,
            imgsz=[3, 48, 320],
            device="gpu",
            save_dir="runs/rec_exp1",
        )
    """

    def __init__(self, config: str | Path):
        self.config = Path(config)

        if not self.config.exists():
            raise FileNotFoundError(f"Config not found: {self.config}")

    def train(
        self,
        *,
        pretrained_model: str | Path | None = None,
        save_dir: str | Path = "runs/train",
        device: str = "gpu",
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
        data_dir: str | Path | None = None,
        train_labels: str | Path | None = None,
        val_labels: str | Path | None = None,
        character_dict_path: str | Path | None = None,
        use_amp: bool | None = None,
        seed: int = 1024,
        extra_overrides: dict[str, Any] | None = None,
    ):
        args = [
            "train.py",
            "-c",
            str(self.config),
        ]

        overrides: dict[str, Any] = {
            "Global.use_gpu": device == "gpu",
            "Global.save_model_dir": str(save_dir),
            "Global.seed": seed,
        }

        if pretrained_model is not None:
            overrides["Global.pretrained_model"] = str(pretrained_model)

        if epochs is not None:
            overrides["Global.epoch_num"] = epochs

        if batch_size is not None:
            overrides["Train.loader.batch_size_per_card"] = batch_size

        if lr is not None:
            overrides["Optimizer.lr.learning_rate"] = lr

        if data_dir is not None:
            overrides["Train.dataset.data_dir"] = str(data_dir)
            overrides["Eval.dataset.data_dir"] = str(data_dir)

        if train_labels is not None:
            overrides["Train.dataset.label_file_list"] = [str(train_labels)]

        if val_labels is not None:
            overrides["Eval.dataset.label_file_list"] = [str(val_labels)]

        if character_dict_path is not None:
            overrides["Global.character_dict_path"] = str(character_dict_path)

        if use_amp is not None:
            overrides["Global.use_amp"] = use_amp

        if extra_overrides:
            overrides.update(extra_overrides)

        for key, value in overrides.items():
            args.extend(["-o", f"{key}={value}"])

        with patched_argv(args):
            config, paddle_device, logger, vdl_writer = program.preprocess(
                is_train=True
            )
            set_seed(config["Global"].get("seed", seed))
            paddleocr_train_main(config, paddle_device, logger, vdl_writer)

        return self

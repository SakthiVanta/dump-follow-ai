"""
YOLO fine-tuning trainer using Ultralytics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from robot.config import TrainingConfig
from robot.logger import logger


class YOLOTrainer:
    """
    Fine-tunes a YOLOv8/YOLO11 model on a custom dataset.

    Parameters
    ----------
    config: TrainingConfig
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self._cfg = config or TrainingConfig()

    def train(
        self,
        data_yaml: str | Path,
        run_name: str = "robot_person_detector",
        resume: bool = False,
    ) -> Path:
        """
        Launch training.

        Parameters
        ----------
        data_yaml:  Path to the data.yaml file produced by DatasetBuilder.
        run_name:   Experiment name for Ultralytics runs/ directory.
        resume:     Resume a previous interrupted training.

        Returns
        -------
        Path to the best weights file.
        """
        from ultralytics import YOLO

        logger.info(f"Training {self._cfg.model_base} on {data_yaml}")

        model = YOLO(self._cfg.model_base)
        results = model.train(
            data=str(data_yaml),
            epochs=self._cfg.epochs,
            batch=self._cfg.batch_size,
            imgsz=self._cfg.image_size,
            name=run_name,
            project=self._cfg.output_dir,
            augment=self._cfg.augment,
            resume=resume,
            verbose=True,
        )

        best_weights = Path(self._cfg.output_dir) / run_name / "weights" / "best.pt"
        logger.info(f"Training complete. Best weights: {best_weights}")
        return best_weights

    def validate(self, weights_path: str | Path, data_yaml: str | Path) -> dict:
        """Run validation and return metrics dict."""
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        metrics = model.val(data=str(data_yaml), verbose=True)
        return {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
        }

    def export(
        self,
        weights_path: str | Path,
        format: str = "onnx",
        imgsz: Optional[int] = None,
    ) -> Path:
        """
        Export model to deployment format.

        Formats: onnx, tflite, ncnn, engine (TensorRT)
        """
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        export_args = {"format": format}
        if imgsz:
            export_args["imgsz"] = imgsz
        exported = model.export(**export_args)
        logger.info(f"Model exported to {format}: {exported}")
        return Path(exported)

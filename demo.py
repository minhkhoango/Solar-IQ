from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO  # type: ignore
from ultralytics.engine.results import Results  # type: ignore
import numpy as np
import logging
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


def predict_and_draw(model_path: Path, image_path: Path, output_path: Path) -> None:
    """
    Runs inference with a YOLOv8 model and draws the resulting bounding boxes
    on the image.

    Args:
        model_path: Path to the trained YOLOv8 model (.pt or .engine).
        image_path: Path to the input image file.
        output_path: Path to save the output image with detections.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load the model. YOLO automatically handles .pt and .engine files.
    logger.info(f">> Loading model from {model_path}...")
    model = YOLO(model_path)

    # Run inference on the image
    logger.info(f">> Running inference on {image_path}...")
    results: List[Results] = model(image_path) # type: ignore

    logging.info(f"results: {type(results)}")

    # It's good practice to get the first result from the list
    if not results or len(results) == 0:
        logger.error(">> No results returned from model.")
        return

    result = results[1]
    image: np.ndarray = result.plot()  # .plot() returns an annotated BGR np.ndarray

    # Save the annotated image
    logger.info(f">> Saving annotated image to {output_path}...")
    cv2.imwrite(str(output_path), image)
    logger.info("✅ Done.")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference and save the annotated image."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/solar_bb_train/weights/best.engine"),
        help="Path to the trained YOLO model (.pt or .engine file).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.jpg"),
        help="Path to save the output annotated image.",
    )
    args = parser.parse_args()

    predict_and_draw(args.model, args.image, args.output)


if __name__ == "__main__":
    main()

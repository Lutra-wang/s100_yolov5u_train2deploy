#!/usr/bin/env python3
"""Run the official Model Zoo detector on a custom COCO subset.

`rdk_yolo_app.py --mode coco2017` correctly writes COCO-style `info` records,
but intentionally rejects any source whose size is not 5,000. This adapter
uses the same official Ultralytics_YOLO_Detect class and its preprocessing,
BPU call, decoding and NMS; it only repeats the official loop for a selected
subset and writes the unchanged `info` records.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-zoo-yolo-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-thres", type=float, default=0.25)
    parser.add_argument("--nms-thres", type=float, default=0.7)
    args = parser.parse_args()

    yolo_dir = Path(args.model_zoo_yolo_dir).resolve()
    sys.path.insert(0, str(yolo_dir.parent))
    # Import the official entry module first. Its dependency guards install
    # tqdm/scipy/numpy/opencv on a stock board before loading the model class.
    from py import rdk_yolo_app
    Ultralytics_YOLO_Detect = rdk_yolo_app.Ultralytics_YOLO_Detect

    detector = Ultralytics_YOLO_Detect(
        model_path=args.model_path,
        classes_num=80,
        nms_thres=args.nms_thres,
        score_thres=args.score_thres,
        reg=16,
        strides=[8, 16, 32],
    )
    image_paths = sorted(Path(args.source).glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No jpg images in {args.source}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for image_path in image_paths:
            result = detector(str(image_path), img_id=int(image_path.stem), draw=False)
            handle.write(result["info"])
    print(f"images={len(image_paths)} output={output}")


if __name__ == "__main__":
    main()

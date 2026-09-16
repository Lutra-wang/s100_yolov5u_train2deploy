#!/usr/bin/env python3
"""Bridge official Model Zoo BPU text output to unmodified COCOeval.

The Model Zoo Python detector owns preprocessing, BPU execution, decoding and
NMS. This helper only converts its documented eight-column `info` output into
COCO JSON and invokes Microsoft's pycocotools, which is the metric route used
by the Model Zoo accuracy table.
"""

import argparse
import json
from pathlib import Path


def category_map_from_annotations(annotations_path):
    with Path(annotations_path).open(encoding="utf-8") as handle:
        categories = json.load(handle)["categories"]
    return {index: category["id"] for index, category in enumerate(sorted(categories, key=lambda x: x["id"]))}


def bpu_line_to_coco(line, category_by_yolo):
    fields = line.strip().split()
    if len(fields) != 8:
        raise ValueError(f"Expected 8 BPU fields, got {len(fields)}: {line!r}")
    image_id, _, yolo_class, score, x1, y1, x2, y2 = fields
    yolo_class = int(yolo_class)
    if yolo_class not in category_by_yolo:
        raise ValueError(f"Unknown YOLO class id: {yolo_class}")
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid xyxy box: {line!r}")
    return {
        "image_id": int(image_id),
        "category_id": category_by_yolo[yolo_class],
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "score": float(score),
    }


def bpu_text_to_json(bpu_text_path, annotations_path, output_path):
    category_by_yolo = category_map_from_annotations(annotations_path)
    detections = []
    for line in Path(bpu_text_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            detections.append(bpu_line_to_coco(line, category_by_yolo))
    Path(output_path).write_text(json.dumps(detections, indent=2), encoding="utf-8")
    return detections


def fp32_to_json(weights, images_dir, annotations_path, output_path, imgsz, conf, iou):
    from ultralytics import YOLO

    category_by_yolo = category_map_from_annotations(annotations_path)
    model = YOLO(weights)
    detections = []
    for image_path in sorted(Path(images_dir).glob("*.jpg")):
        image_id = int(image_path.stem)
        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, iou=iou, device="cpu", verbose=False)[0]
        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for box, class_id, score in zip(xyxy, classes, scores):
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "image_id": image_id,
                "category_id": category_by_yolo[int(class_id)],
                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                "score": float(score),
            })
    Path(output_path).write_text(json.dumps(detections, indent=2), encoding="utf-8")
    return detections


def metric_summary(stats, image_count, detection_count, map50_90):
    return {
        "bbox_all_map_50_95": float(stats[0]),
        "map50": float(stats[1]),
        "bbox_small_map_50_95": float(stats[3]),
        "bbox_medium_map_50_95": float(stats[4]),
        "bbox_large_map_50_95": float(stats[5]),
        "map50_90": float(map50_90),
        "image_count": image_count,
        "detection_count": detection_count,
    }


def evaluate(annotations_path, predictions_path, metrics_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    coco_gt = COCO(str(annotations_path))
    with Path(predictions_path).open(encoding="utf-8") as handle:
        predictions = json.load(handle)
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    evaluator_50_90 = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator_50_90.params.imgIds = sorted(coco_gt.getImgIds())
    evaluator_50_90.params.iouThrs = np.arange(0.50, 0.91, 0.05)
    evaluator_50_90.evaluate()
    evaluator_50_90.accumulate()
    evaluator_50_90.summarize()
    metrics = metric_summary(evaluator.stats, len(evaluator.params.imgIds), len(predictions), evaluator_50_90.stats[0])
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--annotations", required=True)
    common.add_argument("--output", required=True)
    bpu = subparsers.add_parser("bpu-to-json", parents=[common])
    bpu.add_argument("--bpu-text", required=True)
    fp32 = subparsers.add_parser("fp32-to-json", parents=[common])
    fp32.add_argument("--weights", required=True)
    fp32.add_argument("--images-dir", required=True)
    fp32.add_argument("--imgsz", type=int, default=640)
    fp32.add_argument("--conf", type=float, default=0.25)
    fp32.add_argument("--iou", type=float, default=0.7)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--annotations", required=True)
    evaluate_parser.add_argument("--predictions", required=True)
    evaluate_parser.add_argument("--metrics", required=True)
    args = parser.parse_args()
    if args.command == "bpu-to-json":
        bpu_text_to_json(args.bpu_text, args.annotations, args.output)
    elif args.command == "fp32-to-json":
        fp32_to_json(args.weights, args.images_dir, args.annotations, args.output, args.imgsz, args.conf, args.iou)
    else:
        evaluate(args.annotations, args.predictions, args.metrics)


if __name__ == "__main__":
    main()

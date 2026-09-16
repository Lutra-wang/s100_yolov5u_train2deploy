#!/usr/bin/env python3
"""Build the reproducible COCO 2017 val128 dataset used by this case.

This is intentionally a data-preparation helper. OE and Model Zoo do not
provide a script for selecting a custom 128-image COCO subset; model export,
compilation and BPU inference remain in their official scripts.
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


SEED = 20260915
SPLIT_SIZES = {"train": 96, "val": 16, "test": 16}


def build_category_map(categories):
    """Map COCO category ids to the contiguous YOLO class ids."""
    return {category["id"]: index for index, category in enumerate(sorted(categories, key=lambda x: x["id"]))}


def split_image_ids(image_ids):
    """Return a deterministic 96/16/16 split for exactly 128 image ids."""
    if len(image_ids) != sum(SPLIT_SIZES.values()):
        raise ValueError("Exactly 128 image ids are required")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Image ids must be unique")
    shuffled = list(sorted(image_ids))
    random.Random(SEED).shuffle(shuffled)
    train_end = SPLIT_SIZES["train"]
    val_end = train_end + SPLIT_SIZES["val"]
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def yolo_label(annotation, image, category_map):
    x, y, width, height = annotation["bbox"]
    center_x = (x + width / 2) / image["width"]
    center_y = (y + height / 2) / image["height"]
    return f"{category_map[annotation['category_id']]} {center_x:.8f} {center_y:.8f} {width / image['width']:.8f} {height / image['height']:.8f}\n"


def write_split(output, split_name, image_ids, images_by_id, annotations_by_image, category_map, source_images):
    image_dir = output / "images" / split_name
    label_dir = output / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        image = images_by_id[image_id]
        source = source_images / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing COCO source image: {source}")
        target = image_dir / image["file_name"]
        shutil.copy2(source, target)
        label_path = label_dir / f"{Path(image['file_name']).stem}.txt"
        lines = [yolo_label(annotation, image, category_map) for annotation in annotations_by_image[image_id]]
        label_path.write_text("".join(lines), encoding="utf-8")


def build_dataset(annotations_path, images_dir, output):
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)
    output = Path(output)
    with annotations_path.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    images_by_id = {image["id"]: image for image in coco["images"]}
    selected_ids = sorted(images_by_id)[:128]
    splits = split_image_ids(selected_ids)
    category_map = build_category_map(coco["categories"])
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["image_id"] in images_by_id and annotation["category_id"] in category_map:
            annotations_by_image[annotation["image_id"]].append(annotation)

    output.mkdir(parents=True, exist_ok=True)
    for split_name, ids in splits.items():
        write_split(output, split_name, ids, images_by_id, annotations_by_image, category_map, images_dir)

    selected_images = [images_by_id[image_id] for image_id in selected_ids]
    selected_annotations = [annotation for annotation in coco["annotations"] if annotation["image_id"] in set(selected_ids)]
    test_ids = set(splits["test"])
    test_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": [images_by_id[image_id] for image_id in splits["test"]],
        "annotations": [annotation for annotation in selected_annotations if annotation["image_id"] in test_ids],
    }
    (output / "annotations").mkdir(exist_ok=True)
    (output / "annotations" / "instances_test.json").write_text(json.dumps(test_coco, indent=2), encoding="utf-8")
    selection = {
        "name": "coco2017_val128_s100",
        "source": "COCO 2017 val2017",
        "selection_rule": "ascending COCO val2017 image ids, first 128",
        "seed": SEED,
        "split_sizes": SPLIT_SIZES,
        "splits": splits,
        "image_ids": selected_ids,
        "category_id_to_yolo_id": category_map,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    names = {index: category["name"] for index, category in enumerate(sorted(coco["categories"], key=lambda x: x["id"]))}
    yaml = "path: " + str(output.resolve()) + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {index}: {name}\n" for index, name in names.items())
    (output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    return selection


def verify_dataset(output):
    output = Path(output)
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    expected = SPLIT_SIZES
    found = {}
    all_stems = []
    for split_name, expected_count in expected.items():
        images = sorted((output / "images" / split_name).glob("*.jpg"))
        labels = sorted((output / "labels" / split_name).glob("*.txt"))
        if len(images) != expected_count or len(labels) != expected_count:
            raise ValueError(f"{split_name}: expected {expected_count} images and labels, got {len(images)} and {len(labels)}")
        found[split_name] = len(images)
        all_stems.extend(image.stem for image in images)
    if len(all_stems) != 128 or len(set(all_stems)) != 128:
        raise ValueError("Dataset images are not 128 unique files")
    if len(selection["image_ids"]) != 128 or len(set(selection["image_ids"])) != 128:
        raise ValueError("selection.json does not contain 128 unique source ids")
    if not (output / "annotations" / "instances_test.json").is_file():
        raise FileNotFoundError("Missing test COCO annotations")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", help="COCO instances_val2017.json")
    parser.add_argument("--images-dir", help="COCO val2017 directory")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--verify", action="store_true", help="Verify an already-built dataset")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_dataset(args.output), sort_keys=True))
        return
    if not args.annotations or not args.images_dir:
        parser.error("--annotations and --images-dir are required unless --verify is used")
    selection = build_dataset(args.annotations, args.images_dir, args.output)
    print(json.dumps({"name": selection["name"], "split_sizes": selection["split_sizes"]}, sort_keys=True))


if __name__ == "__main__":
    main()

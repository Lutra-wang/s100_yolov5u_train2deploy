import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_coco2017_val128_s100.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildDatasetTests(unittest.TestCase):
    def test_fixed_split_is_complete_disjoint_and_deterministic(self):
        builder = load_builder()
        image_ids = list(range(1, 129))
        first = builder.split_image_ids(image_ids)
        second = builder.split_image_ids(image_ids)

        self.assertEqual(first, second)
        self.assertEqual({name: len(ids) for name, ids in first.items()}, {
            "train": 96,
            "val": 16,
            "test": 16,
        })
        all_ids = first["train"] + first["val"] + first["test"]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(all_ids), 128)
        self.assertEqual(sorted(all_ids), image_ids)


    def test_coco_categories_map_to_contiguous_yolo_indices(self):
        builder = load_builder()
        categories = [{"id": 1}, {"id": 3}, {"id": 90}]
        self.assertEqual(builder.build_category_map(categories), {1: 0, 3: 1, 90: 2})

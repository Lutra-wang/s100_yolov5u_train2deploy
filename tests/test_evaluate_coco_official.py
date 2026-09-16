import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_coco_official.py"


def load_module():
    spec = importlib.util.spec_from_file_location("coco_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CocoBridgeTests(unittest.TestCase):
    def test_official_bpu_line_converts_to_coco_detection(self):
        module = load_module()
        detection = module.bpu_line_to_coco("42\t0\t2\t0.87\t10.00\t20.00\t50.00\t80.00", {2: 3})
        self.assertEqual(detection["image_id"], 42)
        self.assertEqual(detection["category_id"], 3)
        self.assertEqual(detection["bbox"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(detection["score"], 0.87)

    def test_invalid_bpu_line_is_rejected(self):
        module = load_module()
        with self.assertRaises(ValueError):
            module.bpu_line_to_coco("42\t0\t2", {2: 3})

    def test_metric_summary_reports_map50_and_map50_90(self):
        module = load_module()
        stats = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66]
        summary = module.metric_summary(stats, 16, 92, 0.31)
        self.assertEqual(summary["map50"], 0.22)
        self.assertEqual(summary["map50_90"], 0.31)
        self.assertEqual(summary["bbox_all_map_50_95"], 0.11)

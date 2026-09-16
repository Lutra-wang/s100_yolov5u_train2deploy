from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
MANUAL = ROOT / "docs" / "RDK_S100_YOLOv5u_从训练到部署.md"


class ManualStructureTests(unittest.TestCase):
    def setUp(self):
        self.text = MANUAL.read_text(encoding="utf-8")

    def test_stage_zero_to_two_headings_are_present(self):
        headings = [
            "## Stage 0：项目与基础环境",
            "### 0.1 获取项目",
            "### 0.2 配置 OpenExplorer",
            "## Stage 1：模型训练与导出",
            "### 1.1 配置训练环境",
            "### 1.2 准备项目数据集",
            "### 1.3 训练 YOLOv5nu",
            "### 1.4 建立 FP32 精度基线",
            "### 1.5 导出 ONNX",
            "## Stage 2：模型量化与编译",
            "### 2.1 配置量化环境",
            "### 2.2 准备校准集",
            "### 2.3 使用 mapper.py 完成 PTQ 与编译",
            "### 2.4 查看量化结果",
            "### 2.5 RoboGo 量化",
        ]
        for heading in headings:
            with self.subTest(heading=heading):
                self.assertIn(heading, self.text)

    def test_environment_entry_is_not_repeated(self):
        self.assertNotIn("conda run", self.text)
        self.assertNotIn("docker run", self.text)
        self.assertEqual(self.text.count("conda activate oe-s100-yolov5u"), 1)
        self.assertEqual(self.text.count('bash run_docker.sh "$PROJECT_ROOT" cpu'), 1)

    def test_model_name_and_six_outputs_are_consistent(self):
        self.assertIn("yolov5nu_s100_nv12.hbm", self.text)
        self.assertNotIn("yolov5nu_coco2017_val128_s100_nashe_640x640_nv12.hbm", self.text)
        for node in ("output0", "371", "379", "387", "395", "403"):
            self.assertIn(f"`{node}`", self.text)

    def test_architecture_image_exists_and_is_linked(self):
        image = ROOT / "docs" / "images" / "S0-01-project-architecture.jpg"
        self.assertTrue(image.is_file())
        self.assertIn("![项目架构图](images/S0-01-project-architecture.jpg)", self.text)

    def test_dataset_path_is_adapted_to_current_clone(self):
        self.assertIn(
            'sed -i "s|^path:.*|path: $DATASET_ROOT|" '
            '"$DATASET_ROOT/dataset.yaml"',
            self.text,
        )
        self.assertNotIn("/home/lutra/OE_S100", self.text)

    def test_training_uses_fixed_run_directory(self):
        self.assertIn("exist_ok=True", self.text)
        self.assertIn('RUN="$TRAIN_ROOT/runs/coco2017_val128_s100"', self.text)
        self.assertNotIn("| sort | tail -1", self.text)


if __name__ == "__main__":
    unittest.main()

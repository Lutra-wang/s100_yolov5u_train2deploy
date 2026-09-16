from pathlib import Path
import os
import signal
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_with_bpu_monitor.sh"


class BpuMonitorTests(unittest.TestCase):
    def run_wrapper(self, command, ratio="37", interval="0.05"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ratio_file = tmp_path / "ratio"
            ratio_file.write_text(f"{ratio}\n", encoding="utf-8")
            log = tmp_path / "combined.log"
            env = os.environ.copy()
            env["BPU_RATIO_FILE"] = str(ratio_file)
            result = subprocess.run(
                ["bash", str(SCRIPT), "--log", str(log), "--interval", interval, "--", *command],
                text=True,
                capture_output=True,
                env=env,
            )
            return result, log.read_text(encoding="utf-8") if log.exists() else ""

    def test_merges_inference_output_samples_and_summary(self):
        result, log = self.run_wrapper(
            ["bash", "-c", "echo inference-start; sleep 0.18; echo inference-done"]
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("inference-start", log)
        self.assertIn("inference-done", log)
        self.assertRegex(log, r"\[BPU\] ratio=37%")
        self.assertRegex(log, r"\[BPU_SUMMARY\] samples=\d+ average=37\.0% peak=37%")

    def test_returns_wrapped_command_status(self):
        result, log = self.run_wrapper(["bash", "-c", "echo failed-command; sleep 0.08; exit 7"])
        self.assertEqual(result.returncode, 7)
        self.assertIn("failed-command", log)

    def test_rejects_missing_ratio_file_before_running_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "combined.log"
            env = os.environ.copy()
            env["BPU_RATIO_FILE"] = str(Path(tmp) / "missing")
            result = subprocess.run(
                ["bash", str(SCRIPT), "--log", str(log), "--", "bash", "-c", "echo should-not-run"],
                text=True,
                capture_output=True,
                env=env,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("BPU ratio file is not readable", result.stderr)
            self.assertNotIn("should-not-run", result.stdout)

    def test_excludes_spoofed_bpu_lines_from_summary(self):
        result, log = self.run_wrapper(
            ["bash", "-c", "echo '[BPU] ratio=99%'; sleep 0.12"]
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("[BPU] ratio=99%", log)
        self.assertRegex(log, r"\[BPU_SUMMARY\] samples=\d+ average=37\.0% peak=37%")

    def test_rejects_missing_operands_and_invalid_interval(self):
        for args in (("--log",), ("--interval", "0"), ("--interval", "nope")):
            result = subprocess.run(
                ["bash", str(SCRIPT), *args], text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 2, args)
            self.assertIn("Usage:", result.stderr, args)

    def test_terminating_wrapper_cleans_up_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ratio_file = tmp_path / "ratio"
            ratio_file.write_text("37\n", encoding="utf-8")
            log = tmp_path / "combined.log"
            pid_file = tmp_path / "child.pid"
            env = os.environ.copy()
            env["BPU_RATIO_FILE"] = str(ratio_file)
            process = subprocess.Popen(
                [
                    "bash", str(SCRIPT), "--log", str(log), "--interval", "0.02", "--",
                    "bash", "-c", f"echo $$ > {pid_file}; sleep 10",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
            try:
                import time
                time.sleep(0.15)
                child_pid = int(pid_file.read_text(encoding="utf-8").strip())
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2)
                process.communicate(timeout=1)
                self.assertNotEqual(process.returncode, 0)
                time.sleep(0.15)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if process.poll() is None:
                    process.kill()


if __name__ == "__main__":
    unittest.main()

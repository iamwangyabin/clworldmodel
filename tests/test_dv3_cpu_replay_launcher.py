"""Storage-only DV3 launch contracts; no environment or optimizer updates."""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DV3CPUReplayLauncherTests(unittest.TestCase):
    def test_periodic_evaluation_does_not_require_baseline_task_identity(self):
        tree = ast.parse((ROOT / "third_party/arrow/Code/ARROW_and_DV3/Atari/train.py").read_text())
        expressions = [kw.value for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                       and node.func.id == "_evaluate_policy_tasks"
                       for kw in node.keywords if kw.arg == "eligible_task_count"
                       and any(isinstance(child, ast.Name) and child.id == "current_task_id"
                               for child in ast.walk(kw.value))]
        self.assertEqual(len(expressions), 1)
        expression = compile(ast.Expression(expressions[0]), "periodic_eligibility", "eval")
        for task_id, expected in ((None, None), (0, 1), (5, 6)):
            self.assertEqual(eval(expression, {"current_task_id": task_id}), expected)

    def dry_run(self, *extra):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_dv3_fifo_atari.py"),
                 "--dry-run", "--output-dir", str(output), *extra],
                cwd=ROOT, check=True, capture_output=True, text=True,
            )
            self.assertFalse(output.exists())
            return json.loads(result.stdout.split("\ncommand:", 1)[0])

    def test_cpu_changes_only_storage_and_preserves_full_budget(self):
        original = self.dry_run()
        cpu = self.dry_run("--replay-device", "cpu", "--cpu-threads", "8")
        expected = original["resolved_training_config"]
        expected["replay_buffers"][0]["rb_device"] = "cpu"
        expected["replay_observation_dtype"] = "float32"
        self.assertEqual(expected, cpu["resolved_training_config"])
        self.assertEqual(cpu["fifo_slots"], 1024)
        self.assertEqual(cpu["ltdm_slots"], 0)
        self.assertEqual(cpu["sequence_length"], 512)
        self.assertEqual(cpu["replay_storage_budget"]["observation_bytes"], 25769803776)
        self.assertEqual(cpu["replay_storage_budget"]["allocated_tensor_bytes"], 25813843968)
        self.assertEqual(cpu["budgets"]["world_model_updates"], 541000)
        self.assertEqual(cpu["budgets"]["actor_critic_updates"], 432800)
        self.assertEqual(cpu["analysis_snapshot_semantics"], original["analysis_snapshot_semantics"])
        self.assertEqual(cpu["project_pythonpath_prepend"], str(ROOT / "src"))
        self.assertTrue(cpu["command"][cpu["command"].index("--config") + 1].endswith("resolved_training_config.json"))
        self.assertEqual(original["replay_execution_profile"]["device"], "cuda")

    def test_seed_mapping(self):
        for index, seed in ((1, 1337), (2, 31337), (3, 42)):
            launch = self.dry_run("--seed", str(index), "--replay-device", "cpu")
            self.assertEqual(launch["seed"], seed)
            self.assertEqual(launch["resolved_training_config"]["seed"], seed)


if __name__ == "__main__":
    unittest.main()

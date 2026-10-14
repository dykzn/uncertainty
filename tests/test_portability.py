"""Check shard coverage and timing-window accounting without a GPU."""
import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import start_formal
from start_formal import partition
from manage_formal_gpu import stage_rate
from formal_common import ranking_metrics


class PortabilityChecks(unittest.TestCase):
    def test_small_batch_preserves_task_count_and_relative_shards(self):
        class Child:
            pid = 12345
        cards = [{"index": i, "uuid": f"GPU-fixture-{i}", "egl": i} for i in (0, 1)]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "batch"
            command = ["start_formal.py", "--gpus", "0,1", "--output", str(output),
                       "--initial-count", "3", "--task-end", "1", "--suites", "libero_spatial", "--stride", "5"]
            with patch.object(sys, "argv", command), patch.object(start_formal, "detect", return_value=cards), \
                 patch.object(start_formal, "configure_runtime"), patch.object(start_formal.subprocess, "Popen", return_value=Child()):
                start_formal.main()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["expected_episodes"], 6)
            self.assertEqual(manifest["task_end"], 1)
            self.assertEqual(manifest["score_stride"], 5)
            self.assertEqual([l["initials"] for l in manifest["lanes"]], [[0, 1], [2]])
            self.assertEqual([l["jobs"][0]["expected_episodes"] for l in manifest["lanes"]], [4, 2])
            self.assertTrue(all(not Path(j["run"]).is_absolute() for l in manifest["lanes"] for j in l["jobs"]))

    def test_shards_cover_initial_states_without_overlap(self):
        for cards in (1, 2, 3, 4, 8):
            lanes = partition(50, cards)
            flattened = [i for lane in lanes for i in lane]
            self.assertEqual(flattened, list(range(50)))
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertLessEqual(max(map(len, lanes)) - min(map(len, lanes)), 1)

    def test_timing_excludes_paused_and_cross_episode_intervals(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "runtime.json").write_text(json.dumps({"started_at": "2026-01-01T00:00:00+00:00"}))
            (run / "plan.json").write_text(json.dumps({"score_stride": 5}))
            base = {"phase": "running", "split": "standard", "task_id": 1, "episode_idx": 1}
            rows = [{**base, "actions": 0, "elapsed_sec": 0},
                    {**base, "actions": 5, "elapsed_sec": 100},
                    {**base, "actions": 10, "elapsed_sec": 102},
                    {**base, "actions": 15, "elapsed_sec": 104},
                    {**base, "episode_idx": 2, "actions": 20, "elapsed_sec": 106}]
            (run / "worker.log").write_text("\n".join(json.dumps(r) for r in rows))
            origin = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            result = stage_rate(run, origin + 100, origin + 107)
            self.assertEqual(result["intervals"], 2)
            self.assertEqual(result["actions_per_second"], 2.5)

    def test_metrics_do_not_fabricate_single_class_results(self):
        self.assertIsNone(ranking_metrics([False, False], [0.1, 0.5]))
        self.assertIsNone(ranking_metrics([True, True], [0.1, 0.5]))
        self.assertEqual(ranking_metrics([True, False], [0.5, 0.5])["AUROC"], 0.5)


if __name__ == "__main__":
    unittest.main()

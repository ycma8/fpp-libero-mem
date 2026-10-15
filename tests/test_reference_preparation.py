"""Reference split preparation excludes test IDs before feature extraction."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from scripts.prepare_cache import main


class ReferencePreparationTests(unittest.TestCase):
    def test_reference_membership_and_zero_gripper_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            membership = []
            tasks = []
            for task in range(1, 11):
                name = f"KITCHEN_SCENE1_{task}_test_demo.hdf5"
                with h5py.File(root / name, "w") as handle:
                    for index in range(22):
                        group = handle.create_group(f"data/demo_{index}")
                        group.create_dataset(
                            "actions", data=np.zeros((1, 7), dtype=np.float32)
                        )
                        group.create_dataset(
                            "obs/agentview_rgb",
                            data=np.zeros((1, 2, 2, 3), dtype=np.uint8),
                        )
                membership.extend(
                    [
                        {"task": task, "episode": "demo_0", "split": "train"},
                        {"task": task, "episode": "demo_1", "split": "val"},
                    ]
                )
                tasks.append(
                    {
                        "task": task,
                        "file": name,
                        "test_ids": ["demo_21"],
                        "raw_episodes": [
                            {"id": "demo_21", "init_sha": "hash", "horizon": 200}
                        ],
                    }
                )
            reference = root / "split.json"
            reference.write_text(json.dumps({"episodes": membership, "tasks": tasks}))
            np.save(root / "text.npy", np.zeros((10, 512), dtype=np.float32))
            (root / "dino").write_bytes(b"test-checkpoint")
            fake = SimpleNamespace()
            fake.eval = lambda: fake
            fake.to = lambda device: fake
            args = SimpleNamespace(
                raw_glob=[str(root / "*.hdf5")],
                batch=1,
                history_stride=10,
                max_observations=144,
                min_scale=1e-4,
                text_features=root / "text.npy",
                output=root / "output",
                device="cpu",
                dino_model="test",
                dino_checkpoint=root / "dino",
                text_encoder="test",
                reference_split=reference,
            )
            with patch.dict(
                "sys.modules",
                {"timm": SimpleNamespace(create_model=lambda *a, **k: fake)},
            ), patch(
                "scripts.prepare_cache._features",
                side_effect=lambda model, images, *a: np.zeros(
                    (len(images), 196, 768), dtype=np.float16
                ),
            ):
                main(args)
            prepared = root / "output/prepared"
            manifest = json.loads((prepared / "manifest.json").read_text())
            self.assertEqual(len(manifest["episodes"]), 20)
            self.assertEqual(len(manifest["evaluation_trials"]), 10)
            self.assertEqual(
                [row["task"] for row in manifest["episodes"]],
                [task for task in range(1, 11) for _ in range(2)],
            )
            self.assertFalse((prepared / "task_01/demo_21").exists())
            self.assertEqual(
                float(np.load(prepared / "task_01/demo_0/actions.npy")[0, -1]), -1.0
            )


if __name__ == "__main__":
    unittest.main()

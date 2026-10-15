"""Canonical views preserve episode order and reject incompatible caches."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scripts.materialize_reference_cache import materialize


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    prepared = root / "prepared"
    prepared.mkdir(parents=True)
    np.save(prepared / "text.npy", np.zeros((10, 512), dtype=np.float32))
    episodes = []
    for task in (1, 2):
        cache = f"task_{task:02d}/demo_1"
        directory = prepared / cache
        directory.mkdir(parents=True)
        np.save(directory / "features.npy", np.zeros((2, 196, 768), dtype=np.float16))
        np.save(directory / "actions.npy", np.zeros((2, 7), dtype=np.float32))
        episodes.append(
            {
                "task": task,
                "source": f"task{task}:demo_1",
                "split": "train",
                "length": 2,
                "cache": cache,
            }
        )
    manifest = {
        "schema": "fpp_libero_mem_prepared_v1",
        "camera_keys": ["agentview_rgb"],
        "history_stride": 10,
        "max_observations": 144,
        "normalization": {"center": [0] * 7, "scale": [1] * 7, "constant": [False] * 7},
        "dino_model": "test",
        "dino_checkpoint_sha256": "test-hash",
        "text_features_sha256": hashlib.sha256(
            (prepared / "text.npy").read_bytes()
        ).hexdigest(),
        "zero_gripper_to_open": True,
        "feature_extraction": {
            "layout": "legacy-paired32",
            "feature_batch": 64,
            "resize_device": "cuda",
            "backbone_autocast": "bfloat16",
        },
        "episodes": episodes,
        "evaluation_trials": [
            {
                "task": 1,
                "source": "task1:demo_101",
                "initial_state_sha256": "init",
                "horizon": 200,
            }
        ],
    }
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    reference = copy.deepcopy(manifest)
    reference.pop("feature_extraction")
    reference.pop("zero_gripper_to_open")
    reference["episodes"].reverse()
    for record in reference["episodes"]:
        record["cache"] = f'episodes/task{record["task"]:02d}_demo_1'
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps(reference, indent=3) + "\n")
    return root, ref


def test_canonical_manifest_and_order_without_copying(tmp_path: Path) -> None:
    source, reference = _fixture(tmp_path)
    output = tmp_path / "canonical"
    record = materialize(source, reference, output)
    assert (output / "prepared/manifest.json").read_bytes() == reference.read_bytes()
    features = output / "prepared/episodes/task02_demo_1/features.npy"
    assert features.is_symlink()
    assert features.resolve() == source / "prepared/task_02/demo_1/features.npy"
    assert not record["feature_values_verified_against_reference"]
    with pytest.raises(FileExistsError):
        materialize(source, reference, output)


@pytest.mark.parametrize(
    "mismatch", ["split", "heldout", "normalization", "batch", "path", "dtype"]
)
def test_rejects_incompatible_data(tmp_path: Path, mismatch: str) -> None:
    source, reference = _fixture(tmp_path)
    path = source / "prepared/manifest.json"
    manifest = json.loads(path.read_text())
    if mismatch == "split":
        manifest["episodes"][0]["split"] = "val"
    elif mismatch == "heldout":
        manifest["evaluation_trials"][0]["horizon"] += 1
    elif mismatch == "normalization":
        manifest["normalization"]["scale"][0] = 2
    elif mismatch == "batch":
        manifest["feature_extraction"]["feature_batch"] = 32
    elif mismatch == "path":
        manifest["episodes"][0]["cache"] = "../escape"
    else:
        np.save(
            source / "prepared/task_01/demo_1/features.npy",
            np.zeros((2, 196, 768), dtype=np.float32),
        )
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        materialize(source, reference, tmp_path / "canonical")
    assert not (tmp_path / "canonical").exists()


def test_rejects_reference_traversal(tmp_path: Path) -> None:
    source, reference = _fixture(tmp_path)
    manifest = json.loads(reference.read_text())
    manifest["episodes"][0]["cache"] = "../escape"
    reference.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        materialize(source, reference, tmp_path / "canonical")

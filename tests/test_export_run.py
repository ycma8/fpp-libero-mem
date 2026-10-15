import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fpp_libero_mem.prefix_model import PrefixPatchDiT
from scripts.export_run import _sha, export_run
from scripts.restore_text_features import restore
from test_fresh_evaluation import fixture_run
from test_prefix_model import small_config


def completed_run(root: Path) -> tuple[Path, Path, Path]:
    evaluation = root / "evaluation"
    evaluation.mkdir()
    manifest, rows = fixture_run(evaluation)
    data = root / "data"
    (data / "prepared").mkdir(parents=True)
    torch.manual_seed(17)
    model_config = small_config()
    model_config.update(
        cameras=1,
        feature_dim=768,
        text_dim=512,
        grid_size=14,
        max_steps=144,
        horizon=16,
        temporal_position="rope",
        append_eoe=False,
        training_steps=20000,
    )
    model = PrefixPatchDiT(model_config)
    matrix = data / "prepared/text.npy"
    np.save(matrix, np.arange(5120, dtype=np.float32).reshape(10, 512))
    source = json.loads(manifest.read_text())
    stats = dict(center=[0] * 7, scale=[1] * 7, constant=[False] * 7)
    source.update(
        normalization=stats,
        history_stride=10,
        camera_keys=["agentview_rgb"],
        text_features_sha256=_sha(matrix),
        text_encoder="synthetic test fixture",
    )
    (data / "prepared/manifest.json").write_text(json.dumps(source))
    training = root / "training"
    training.mkdir()
    config = dict(
        model=model_config,
        seed=42,
        data_manifest_sha256=_sha(data / "prepared/manifest.json"),
        training=dict(source_revision="fixture", global_batch=8, world_size=2),
    )
    (training / "config.json").write_text(
        json.dumps({**config, "data_root": "/private/data"})
    )
    checkpoint = dict(
        format="fpp_libero_mem_v1",
        step=20000,
        config=model_config,
        model=model.state_dict(),
        seed=42,
        data_manifest_sha256=config["data_manifest_sha256"],
        training=config["training"],
        optimizer={"state": {0: {"step": torch.tensor(20000.0)}}},
        normalization=stats,
        history_stride=10,
        text_features_sha256=_sha(matrix),
    )
    torch.save(checkpoint, training / "step_020000.pt")
    settings = json.loads((evaluation / "settings.json").read_text())
    settings.update(
        checkpoint_sha256=_sha(training / "step_020000.pt"),
        data_manifest_sha256=config["data_manifest_sha256"],
        protocol="deterministic-xml-reset-full-horizon-v3",
        image_transform="flipud",
        camera_size=256,
        flow_steps=10,
        settle_steps=20,
        simulator_revision="fixture",
        history_stride=10,
        seed_rule="base_seed + 1000 * task + episode_number",
        evaluator_sha256="fixture",
        evaluation_library_sha256="fixture",
        runtime={"torch": torch.__version__},
        raw_glob="/private/demonstrations/*.hdf5",
        save_frames=False,
    )
    (evaluation / "settings.json").write_text(json.dumps(settings))
    for row in rows:
        row["bddl"] = "/private/environment/task.bddl"
    (evaluation / "trials.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (training / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(dict(step=step, loss=loss, grad_norm=1.0, elapsed_s=elapsed))
            for step, loss, elapsed in [(1, 1.0, 2.0), (20000, 0.1, 200.0)]
        )
        + "\n"
    )
    return training, evaluation, data


def test_export_verifies_weights_redacts_paths_and_restores_exact_text(
    tmp_path: Path,
) -> None:
    training, evaluation, data = completed_run(tmp_path)
    out = tmp_path / "public"
    policy = tmp_path / "policy.pt"
    report = export_run(training, evaluation, data, out, policy)
    assert report["verified"] and report["metrics"]["trials"] == 200
    chart = json.loads((out / "results.json").read_text())
    assert len(chart["tasks"]) == 10 and chart["tasks"][0]["subgoals_total"] == 20
    assert chart["tasks"][5]["subgoals_total"] == 140
    assert chart["metrics"]["mean_subgoal_completion"] == 0
    for path in out.glob("*.json*"):
        assert "/private/" not in path.read_text()
    inference = torch.load(policy, weights_only=True)
    assert "optimizer" not in inference and inference["step"] == 20000
    restored = tmp_path / "restored.npy"
    restore(out / "frozen-text.json", restored)
    assert restored.read_bytes() == (data / "prepared/text.npy").read_bytes()
    with pytest.raises(FileExistsError):
        restore(out / "frozen-text.json", restored)


def test_export_rejects_wrong_weights_and_unfinished_training(tmp_path: Path) -> None:
    training, evaluation, data = completed_run(tmp_path)
    checkpoint = training / "step_020000.pt"
    saved = torch.load(checkpoint, weights_only=True)
    saved["config"]["append_eoe"] = True
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="no-EOE"):
        export_run(training, evaluation, data, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    another = tmp_path / "another"
    another.mkdir()
    training, evaluation, data = completed_run(another)
    (training / "metrics.jsonl").write_text(
        json.dumps(dict(step=1, loss=1.0, grad_norm=1.0, elapsed_s=2.0)) + "\n"
    )
    with pytest.raises(ValueError, match="complete 20K"):
        export_run(training, evaluation, data, tmp_path / "unfinished")
    assert not (tmp_path / "unfinished").exists()


def test_restore_rejects_corrupt_payload(tmp_path: Path) -> None:
    training, evaluation, data = completed_run(tmp_path)
    out = tmp_path / "public"
    export_run(training, evaluation, data, out)
    artifact = json.loads((out / "frozen-text.json").read_text())
    artifact["npy_sha256"] = "0" * 64
    (out / "frozen-text.json").write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="checksum"):
        restore(out / "frozen-text.json", tmp_path / "bad.npy")
    assert not (tmp_path / "bad.npy").exists()

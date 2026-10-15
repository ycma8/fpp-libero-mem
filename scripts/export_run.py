"""Export one verified no-EOE 20K run into public records and article data.

Run with ``python -m scripts.export_run`` from the repository root. The optional
inference-only checkpoint must be outside the small public record directory.
Nothing is committed, uploaded, or substituted for a missing final checkpoint.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from scripts.verify_evaluation import GOALS, verify

TARGET_STEP = 20000


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_run(
    training: Path,
    evaluation: Path,
    data_root: Path,
    output: Path,
    policy_output: Path | None = None,
) -> dict:
    """Validate the final weights and full evaluation, then stage public files.

    Returns:
        Verified aggregate metrics and checkpoint fingerprints.

    Raises:
        ValueError: If the checkpoint, training log, data, or results disagree.
        FileExistsError: If an output would overwrite an existing artifact.
    """
    import numpy as np
    import torch
    from fpp_libero_mem.prefix_model import PrefixPatchDiT

    if output.exists() or (policy_output is not None and policy_output.exists()):
        raise FileExistsError("Choose fresh result and policy destinations")
    if policy_output is not None and policy_output.resolve().is_relative_to(
        output.resolve()
    ):
        raise ValueError(
            "Keep the large policy checkpoint outside the public record directory"
        )
    manifest_path = data_root / "prepared/manifest.json"
    verified = verify(evaluation, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((training / "config.json").read_text())
    settings = json.loads((evaluation / "settings.json").read_text())
    rows = [
        json.loads(line)
        for line in (evaluation / "trials.jsonl").read_text().splitlines()
    ]
    metrics = [
        json.loads(line)
        for line in (training / "metrics.jsonl").read_text().splitlines()
    ]
    checkpoint = training / f"step_{TARGET_STEP:06d}.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_sha256 = _sha(checkpoint)
    manifest_sha256 = _sha(manifest_path)
    if (
        saved.get("format") != "fpp_libero_mem_v1"
        or saved.get("step") != TARGET_STEP
        or saved["config"].get("append_eoe") is not False
        or saved["config"].get("training_steps") != TARGET_STEP
    ):
        raise ValueError("Expected the final no-EOE 20K checkpoint")
    for key in ["model", "training", "seed", "data_manifest_sha256"]:
        actual = saved["config"] if key == "model" else saved[key]
        if config[key] != actual:
            raise ValueError(f"Training config/checkpoint mismatch: {key}")
    if (
        config["data_manifest_sha256"] != manifest_sha256
        or settings.get("data_manifest_sha256") != manifest_sha256
        or settings["checkpoint_sha256"] != checkpoint_sha256
    ):
        raise ValueError("Evaluation did not use these exact weights and data manifest")
    if settings.get("protocol") != "deterministic-xml-reset-full-horizon-v3":
        raise ValueError(
            "Expected the independently seeded full-horizon evaluation protocol"
        )
    if (
        saved["normalization"] != manifest["normalization"]
        or saved["history_stride"] != manifest["history_stride"]
    ):
        raise ValueError("Checkpoint preprocessing differs from the prepared data")
    if saved["config"]["cameras"] != len(manifest["camera_keys"]):
        raise ValueError("Checkpoint camera count differs from the data contract")
    if not config["training"].get("source_revision"):
        raise ValueError("The training source revision must be recorded")
    if not metrics or metrics[0]["step"] != 1 or metrics[-1]["step"] != TARGET_STEP:
        raise ValueError("Training metrics do not cover the complete 20K run")
    previous_step, previous_time = 0, -1.0
    for point in metrics:
        if (
            not previous_step < point["step"] <= TARGET_STEP
            or point["elapsed_s"] <= previous_time
            or not all(
                math.isfinite(point[key]) for key in ["loss", "elapsed_s", "grad_norm"]
            )
        ):
            raise ValueError("Non-finite or unordered training metrics")
        previous_step, previous_time = point["step"], point["elapsed_s"]
    states = saved.get("optimizer", {}).get("state", {})
    updates = [float(state["step"]) for state in states.values()]
    if not updates or any(
        not value.is_integer() or int(value) != TARGET_STEP for value in updates
    ):
        raise ValueError("Adam update counters do not match the final checkpoint")
    model = PrefixPatchDiT(saved["config"])
    model.load_state_dict(saved["model"], strict=True)
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    del model
    text_path = data_root / "prepared/text.npy"
    text = np.load(text_path, allow_pickle=False)
    if (
        text.shape != (10, 512)
        or text.dtype != np.float32
        or not np.isfinite(text).all()
    ):
        raise ValueError("Invalid frozen task-conditioning matrix")
    text_sha = _sha(text_path)
    if (
        text_sha != saved["text_features_sha256"]
        or text_sha != manifest["text_features_sha256"]
    ):
        raise ValueError("Frozen text features differ from training")
    tasks = []
    for task in range(1, 11):
        records = [row for row in rows if row["task"] == task]
        successes = sum(row["success"] for row in records)
        completed = sum(row["subgoals_completed"] for row in records)
        tasks.append(
            dict(
                task=task,
                trials=20,
                successes=successes,
                success_rate=successes / 20,
                subgoals_completed=completed,
                subgoals_total=20 * GOALS[task - 1],
                subgoal_completion=completed / (20 * GOALS[task - 1]),
                overshot=sum(row["overshot"] for row in records),
            )
        )
    public_settings = {
        key: settings[key]
        for key in [
            "seed",
            "image_transform",
            "camera_size",
            "flow_steps",
            "settle_steps",
            "simulator_revision",
            "history_stride",
            "protocol",
            "full_trial_count",
            "requested_trial_count",
            "seed_rule",
            "checkpoint_sha256",
            "data_manifest_sha256",
            "evaluator_sha256",
            "evaluation_library_sha256",
            "runtime",
        ]
    }
    public_config = {
        key: config[key]
        for key in ["model", "training", "seed", "data_manifest_sha256"]
    }
    allowed = [
        "task",
        "episode",
        "source",
        "init_sha256",
        "horizon",
        "steps",
        "seed",
        "success",
        "subgoals_completed",
        "subgoals_total",
        "overshot",
        "frames",
    ]
    public_rows = []
    for row in rows:
        public = {key: row[key] for key in allowed if key in row}
        if "bddl" in row:
            public["bddl"] = Path(row["bddl"]).name
        public_rows.append(public)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fpp-export-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "records"
        stage.mkdir()
        (stage / "trials.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in public_rows)
        )
        _write(stage / "settings.json", public_settings)
        _write(stage / "training-config.json", public_config)
        protocol = {
            key: manifest[key]
            for key in [
                "schema",
                "camera_keys",
                "history_stride",
                "max_observations",
                "normalization",
                "text_features_sha256",
                "text_encoder",
                "dino_model",
                "dino_checkpoint_sha256",
                "source_manifest_sha256",
                "evaluation_trials",
            ]
            if key in manifest
        }
        protocol["episodes"] = [
            {
                key: row[key]
                for key in ["task", "source", "split", "length"]
                if key in row
            }
            for row in manifest.get("episodes", [])
        ]
        _write(stage / "data-protocol.json", protocol)
        summary = json.loads((evaluation / "summary.json").read_text())
        _write(stage / "summary.json", summary)
        shutil.copyfile(training / "metrics.jsonl", stage / "training-metrics.jsonl")
        with (stage / "tasks.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tasks[0]))
            writer.writeheader()
            writer.writerows(tasks)
        frames = [frame for row in public_rows for frame in row.get("frames", [])]
        if settings.get("save_frames") and len(frames) != 60:
            raise ValueError("Expected six first-trial snapshots for each task")
        for frame in frames:
            relative = Path(frame)
            original = (evaluation / relative).resolve()
            if relative.is_absolute() or not original.is_relative_to(
                evaluation.resolve()
            ):
                raise ValueError("Snapshot path escapes the evaluation directory")
            if not original.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Invalid PNG snapshot")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        frozen_text = dict(
            format="fpp-frozen-text-npy-base64-v1",
            shape=[10, 512],
            dtype="float32",
            task_order=list(range(1, 11)),
            encoder_description=manifest["text_encoder"],
            encoder_revision=None,
            note="Exact conditioning vectors; the original encoder revision was not recovered.",
            npy_sha256=text_sha,
            npy_base64=base64.b64encode(text_path.read_bytes()).decode(),
        )
        _write(stage / "frozen-text.json", frozen_text)
        chart = dict(
            schema=1,
            complete=True,
            method="FPP",
            checkpoint_updates=TARGET_STEP,
            metrics={key: summary[key] for key in verified if key in summary},
            tasks=tasks,
            training_curve=[dict(step=p["step"], loss=p["loss"]) for p in metrics],
            learnable_parameters=parameters,
            training_loop_hours=metrics[-1]["elapsed_s"] / 3600,
        )
        _write(stage / "results.json", chart)
        evidence = dict(
            schema=1,
            verified=True,
            checkpoint_updates=TARGET_STEP,
            source_training=config["training"]["source_revision"],
            checkpoint_sha256=checkpoint_sha256,
            data_manifest_sha256=manifest_sha256,
            raw_trials_sha256=_sha(evaluation / "trials.jsonl"),
            public_trials_sha256=_sha(stage / "trials.jsonl"),
            path_redaction="Only BDDL paths and non-public run paths are removed; metric fields are retained.",
            learnable_parameters=parameters,
        )
        if policy_output is not None:
            policy_output.parent.mkdir(parents=True, exist_ok=True)
            policy = {key: value for key, value in saved.items() if key != "optimizer"}
            with tempfile.NamedTemporaryFile(
                dir=policy_output.parent, suffix=".pt", delete=False
            ) as stream:
                policy_temporary = Path(stream.name)
            try:
                torch.save(policy, policy_temporary)
                evidence["inference_checkpoint_sha256"] = _sha(policy_temporary)
                evidence["inference_checkpoint_filename"] = policy_output.name
                policy_temporary.chmod(0o664)
                os.link(policy_temporary, policy_output)
            finally:
                policy_temporary.unlink(missing_ok=True)
        _write(stage / "evidence.json", evidence)
        stage.rename(output)
    return dict(verified=True, metrics=chart["metrics"], evidence=evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["training", "evaluation", "data-root", "output"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--policy-output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            export_run(
                args.training,
                args.evaluation,
                args.data_root,
                args.output,
                args.policy_output,
            ),
            indent=2,
        )
    )

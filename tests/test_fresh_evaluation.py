import json
from pathlib import Path

import pytest
from scripts.verify_evaluation import GOALS, verify


def fixture_run(root: Path) -> tuple[Path, list[dict]]:
    rows, plan = [], []
    for task, total in enumerate(GOALS, 1):
        for episode in range(20):
            key = f"demo_{episode + 100}"
            row = dict(
                task=task,
                episode=key,
                source=f"task{task}.hdf5",
                init_sha256=f"hash{task}-{episode}",
                steps=200,
                seed=42 + 1000 * task + episode + 100,
                success=False,
                overshot=False,
                subgoals_completed=0,
                subgoals_total=total,
            )
            rows.append(row)
            plan.append(
                dict(
                    task=task,
                    source=f"task{task}.hdf5:{key}",
                    initial_state_sha256=row["init_sha256"],
                    horizon=200,
                )
            )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(dict(evaluation_trials=plan)))
    (root / "settings.json").write_text(
        json.dumps(dict(seed=42, full_trial_count=200, requested_trial_count=200))
    )
    (root / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "summary.json").write_text(
        json.dumps(
            dict(
                trials=200,
                successes=0,
                whole_task_success=0,
                subgoals_completed=0,
                subgoals_total=620,
                mean_subgoal_completion=0,
                pooled_subgoal_completion=0,
                overshot=0,
                tasks=[
                    dict(task=task, trials=20, successes=0, subgoal_completion=0)
                    for task in range(1, 11)
                ],
            )
        )
    )
    return manifest, rows


def test_complete_verification_rejects_duplicate_and_summary_edit(
    tmp_path: Path,
) -> None:
    manifest, rows = fixture_run(tmp_path)
    result = verify(tmp_path, manifest)
    assert result["verified"] and result["subgoals_total"] == 620
    rows[-1] = rows[0]
    (tmp_path / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="membership"):
        verify(tmp_path, manifest)
    manifest, _ = fixture_run(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["whole_task_success"] = 0.5
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Summary"):
        verify(tmp_path, manifest)


def test_smoke_and_wrong_state_are_not_benchmark_results(tmp_path: Path) -> None:
    manifest, rows = fixture_run(tmp_path)
    settings = json.loads((tmp_path / "settings.json").read_text())
    settings["requested_trial_count"] = 1
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="subset"):
        verify(tmp_path, manifest)
    manifest, rows = fixture_run(tmp_path)
    rows[0]["init_sha256"] = "different"
    (tmp_path / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="State"):
        verify(tmp_path, manifest)


def test_nan_and_per_task_edits_are_rejected(tmp_path: Path) -> None:
    manifest, _ = fixture_run(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["mean_subgoal_completion"] = float("nan")
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Summary"):
        verify(tmp_path, manifest)
    manifest, _ = fixture_run(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    summary["tasks"][0]["successes"] = 20
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Per-task"):
        verify(tmp_path, manifest)

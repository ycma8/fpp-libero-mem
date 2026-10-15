"""Recompute archived LIBERO-Mem metrics without a simulator or ML dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition: bool, message: str) -> None:
    """Reject inconsistent evidence even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def summarize(rows: list[dict], checkpoint: str) -> dict:
    """Validate trial records and compute task-mean and pooled metrics."""
    require(len(rows) == 200, "Expected 200 trials per policy")
    require(len({(r["task"], r["episode"]) for r in rows}) == 200, "Duplicate trials")
    require({r["task"] for r in rows} == set(range(1, 11)), "Expected tasks T1--T10")
    subgoals = [1, 1, 3, 3, 5, 7, 3, 4, 2, 2]
    for row in rows:
        require(row["checkpoint"] == checkpoint, "Mixed checkpoint IDs")
        require(row["seed"] == 42 and row["split"] == "test", "Mixed seed/split")
        require(type(row["horizon"]) is int and row["horizon"] > 0, "Invalid horizon")
        for field in ("full_success", "official_final_done", "overshot"):
            require(type(row[field]) is bool, f"Invalid boolean: {field}")
        total, done = row["total_subgoals"], row["completed_subgoals"]
        require(type(total) is int and total == subgoals[row["task"]-1], "Invalid subgoal denominator")
        require(type(done) is int and 0 <= done <= total, "Invalid completed subgoals")
        strict = row["official_final_done"] and done == total and not row["overshot"]
        require(row["full_success"] == strict, "Inconsistent strict success")
        require(math.isclose(row["subgoal_completion"], done/total, abs_tol=1e-12), "Inconsistent subgoal fraction")

    tasks = []
    for task in range(1, 11):
        group = [r for r in rows if r["task"] == task]
        require(len(group) == 20, f"Expected 20 trials for T{task}")
        tasks.append(dict(task=task, trials=len(group),
            successes=sum(r["full_success"] for r in group),
            completed_subgoals=sum(r["completed_subgoals"] for r in group),
            total_subgoals=sum(r["total_subgoals"] for r in group),
            overshot=sum(r["overshot"] for r in group)))
    successes = sum(t["successes"] for t in tasks)
    completed = sum(t["completed_subgoals"] for t in tasks)
    total = sum(t["total_subgoals"] for t in tasks)
    return dict(trials=len(rows), successes=successes, success_rate=successes/len(rows),
        completed_subgoals=completed, total_subgoals=total, subgoal_pooled=completed/total,
        subgoal_task_mean=sum(t["completed_subgoals"]/t["total_subgoals"] for t in tasks)/10,
        overshot=sum(t["overshot"] for t in tasks), tasks=tasks)


def verify(root: Path) -> dict:
    """Check exported hashes, paired trials, and every recorded summary field."""
    checksum = json.loads((root / "checksums.json").read_text())
    for name, expected in checksum["exported_files"].items():
        require(Path(name).name == name, "Invalid evidence filename")
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        require(actual == expected, f"SHA256 mismatch: {name}")
    records, result = {}, {}
    for policy, checkpoint in [("fpp", "fpp-step020000"), ("dp", "dp-step030000")]:
        rows = [json.loads(line) for line in (root / f"{policy}.rollouts.jsonl").read_text().splitlines() if line.strip()]
        result[policy] = summarize(rows, checkpoint)
        records[policy] = {(r["task"], r["episode"]): (r["horizon"], r["split"], r["seed"]) for r in rows}
    require(records["fpp"] == records["dp"], "Task/initial-state ID/horizon/seed pairs differ")
    split = json.loads((root / "split-manifest.json").read_text())
    training = {(r["task"], r["episode"]) for r in split["episodes"] if r["split"] == "train"}
    validation = {(r["task"], r["episode"]) for r in split["episodes"] if r["split"] == "val"}
    test = {(task["task"], key) for task in split["tasks"] for key in task["test_ids"]}
    require(len(training) == 861 and len(validation) == 100 and len(test) == 200,
            "Unexpected historical split counts")
    require(not (training & validation or training & test or validation & test), "Split IDs overlap")
    require(set(records["fpp"]) == test, "Rollout IDs differ from the recovered test split")
    for task in split["tasks"]:
        states = {r["id"]: r for r in task["raw_episodes"]}
        train_hashes = {states[key]["init_sha"] for number, key in training if number == task["task"]}
        test_hashes = {states[key]["init_sha"] for key in task["test_ids"]}
        require(not train_hashes & test_hashes, "Recorded train/test initial-state hashes overlap")
        for key in task["test_ids"]:
            require(records["fpp"][(task["task"], key)][0] == states[key]["horizon"], "Historical horizon mismatch")
    replay = json.loads((root / "replay-audit.json").read_text())
    require(len(replay["results"]) == 30 and sum(r["full_success"] for r in replay["results"]) == 28,
            "Recovered replay counts changed")
    expected = json.loads((root / "summary.json").read_text())
    require(result == expected, "Recomputed metrics differ from summary.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
        default=Path(__file__).resolve().parents[1] / "results/libero-mem-reference")
    args = parser.parse_args()
    metrics = verify(args.results)
    print(json.dumps({"status": "passed", "policies": metrics}, indent=2))

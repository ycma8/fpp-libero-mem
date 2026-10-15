"""Create a canonical cache view without copying or changing model inputs.

This adapter reconciles relative paths, episode ordering and JSON serialization
with the published launch manifest. It checks the semantic data contract and
array headers, but does not certify numerical equality of visual features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {"..", ".", ""} for p in value.split("/")):
        raise ValueError(f"Unsafe relative cache path: {value!r}")
    if "\\" in value:
        raise ValueError("Cache paths must use forward slashes")
    return Path(*path.parts)


def _index(records: list[dict], evaluation: bool = False) -> dict:
    result = {}
    for record in records:
        key = (record["task"], record["source"])
        if key in result:
            raise ValueError(f"Duplicate episode identity: {key}")
        result[key] = (
            record
            if evaluation
            else {key: value for key, value in record.items() if key != "cache"}
        )
    return result


def _check_contract(actual: dict, reference: dict) -> None:
    for key in (
        "schema",
        "camera_keys",
        "history_stride",
        "max_observations",
        "normalization",
        "dino_model",
        "dino_checkpoint_sha256",
        "text_features_sha256",
    ):
        if actual.get(key) != reference.get(key) or key not in reference:
            raise ValueError(f"Data contract mismatch: {key}")
    if _index(actual["episodes"]) != _index(reference["episodes"]):
        raise ValueError("Episode membership, split, length or source differs")
    if _index(actual["evaluation_trials"], True) != _index(
        reference["evaluation_trials"], True
    ):
        raise ValueError("Held-out identities, initial states or horizons differ")
    extraction = actual.get("feature_extraction", {})
    expected = {
        "layout": "legacy-paired32",
        "feature_batch": 64,
        "resize_device": "cuda",
        "backbone_autocast": "bfloat16",
    }
    if extraction != expected:
        raise ValueError("Prepare features with legacy-paired32, CUDA BF16, batch 64")
    if actual.get("zero_gripper_to_open") is not True:
        raise ValueError(
            "The reference preparation requires zero gripper mapped to open"
        )


def materialize(source_root: Path, reference_path: Path, output: Path) -> dict:
    """Validate preparation and create a new cache view using file symlinks.

    Args:
        source_root: Generic preparation output containing ``prepared/``.
        reference_path: Published exact launch manifest.
        output: A fresh data root on the same filesystem as its parent.

    Returns:
        Provenance record for the created view.

    Raises:
        ValueError: Data contracts, paths or array headers do not match.
        FileExistsError: The destination already exists.
    """
    source = (source_root / "prepared").resolve(strict=True)
    source_manifest = source / "manifest.json"
    reference_bytes = reference_path.read_bytes()
    actual = json.loads(source_manifest.read_text())
    reference = json.loads(reference_bytes)
    _check_contract(actual, reference)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    links = []
    actual_records = {(e["task"], e["source"]): e for e in actual["episodes"]}
    seen = set()
    for episode in reference["episodes"]:
        target = _relative_path(episode["cache"])
        if target in seen:
            raise ValueError("Duplicate canonical cache path")
        seen.add(target)
        original = actual_records[episode["task"], episode["source"]]
        directory = (source / _relative_path(original["cache"])).resolve(strict=True)
        if not directory.is_relative_to(source):
            raise ValueError("Source episode escapes its prepared directory")
        for filename, shape, dtype in (
            ("features.npy", (episode["length"], 196, 768), np.dtype("float16")),
            ("actions.npy", (episode["length"], 7), np.dtype("float32")),
        ):
            path = (directory / filename).resolve(strict=True)
            if not path.is_relative_to(source):
                raise ValueError("Source array escapes its prepared directory")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"Invalid array header: {filename} in {target}")
            links.append((target / filename, path))
    text = source / "text.npy"
    matrix = np.load(text, allow_pickle=False)
    if (
        matrix.shape != (10, 512)
        or matrix.dtype != np.float32
        or not np.isfinite(matrix).all()
        or _digest(text) != reference["text_features_sha256"]
    ):
        raise ValueError("Task text matrix differs from the reference")
    record = {
        "schema": "fpp_reference_cache_view_v1",
        "source_manifest_sha256": _digest(source_manifest),
        "reference_manifest_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "episodes": len(reference["episodes"]),
        "link_mode": "absolute_symlinks",
        "feature_values_verified_against_reference": False,
        "feature_extraction": actual["feature_extraction"],
    }
    temporary = Path(tempfile.mkdtemp(prefix=".fpp-cache-view-", dir=output.parent))
    reserved = False
    try:
        prepared = temporary / "prepared"
        prepared.mkdir()
        for relative, original_path in links:
            destination = prepared / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(original_path)
        (prepared / "text.npy").symlink_to(text.resolve())
        (prepared / "manifest.json").write_bytes(reference_bytes)
        (temporary / "cache-view.json").write_text(json.dumps(record, indent=2) + "\n")
        # Reserve exclusively, including against a competing empty directory.
        output.mkdir()
        reserved = True
        (temporary / "prepared").rename(output / "prepared")
        (temporary / "cache-view.json").rename(output / "cache-view.json")
    except BaseException:
        if reserved:
            shutil.rmtree(output)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source, args.reference, args.output), indent=2))

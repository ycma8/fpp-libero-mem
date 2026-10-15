"""Restore the exact frozen task-conditioning matrix supplied with an experiment."""

import argparse
import base64
import hashlib
import json
from pathlib import Path


def restore(artifact: Path, output: Path) -> str:
    """Check the artifact checksum and restore the original NPY bytes.

    Returns:
        SHA-256 of the restored file.

    Raises:
        ValueError: If the artifact format or checksum is invalid.
        FileExistsError: If the destination already exists.
    """
    record = json.loads(artifact.read_text())
    if (
        record.get("format") != "fpp-frozen-text-npy-base64-v1"
        or record.get("shape") != [10, 512]
        or record.get("dtype") != "float32"
    ):
        raise ValueError("Expected the ten-task float32 conditioning artifact")
    payload = base64.b64decode(record["npy_base64"], validate=True)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record["npy_sha256"] or not payload.startswith(b"\x93NUMPY"):
        raise ValueError("Frozen text artifact checksum or NPY header differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"sha256": restore(args.artifact, args.output)}, indent=2))

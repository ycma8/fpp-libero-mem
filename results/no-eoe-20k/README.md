# No-EOE FPP, fixed 20K reproduction

**Historical experiment.** Current defaults have been restored to the EOE
reference model. This directory preserves the separate simplified no-EOE run
without relabeling its result. The commands below describe its recorded source
revision, not the restored defaults; use commit `fbdd3c6` for that training code.
This experiment changed other recipe components and is not an isolated EOE ablation.

This experiment has completed FPP training, without an appended terminal
observation, at the preselected 20,000-update checkpoint, and all 200 held-out
trials. No DP or EOE-control model was trained. The new measured results are
distinct from the archived 50.5% whole-task score.

## Measured results

- Whole-task success: **51/200 = 25.50%**.
- Task-mean subgoal completion: **49.14%**.
- Pooled subgoal completion: **297/620 = 47.90%**.
- Overshot trials: **28/200**.

| Task | Whole-task success | Subgoal completion |
|---|---:|---:|
| T1 | 11/20 (55.00%) | 14/20 (70.00%) |
| T2 | 3/20 (15.00%) | 9/20 (45.00%) |
| T3 | 5/20 (25.00%) | 35/60 (58.33%) |
| T4 | 0/20 (0.00%) | 31/60 (51.67%) |
| T5 | 4/20 (20.00%) | 65/100 (65.00%) |
| T6 | 2/20 (10.00%) | 86/140 (61.43%) |
| T7 | 0/20 (0.00%) | 3/60 (5.00%) |
| T8 | 0/20 (0.00%) | 0/80 (0.00%) |
| T9 | 12/20 (60.00%) | 24/40 (60.00%) |
| T10 | 14/20 (70.00%) | 30/40 (75.00%) |

The [raw trials](trials.jsonl), [per-task counts](tasks.csv),
[settings](settings.json), [summary](summary.json) and [evidence hashes](evidence.json)
are the measurement record. Recompute them from the repository root:

```bash
python scripts/verify_evaluation.py \
  --evaluation results/no-eoe-20k \
  --manifest results/no-eoe-20k/prepared-manifest.json
```

This single-seed run is lower than the archived 50.5% result. Sampling frequency,
normalization, spatial encoding, optimizer schedule and EOE differ between the
recipes; the difference cannot be attributed to EOE alone. Published Table 4
values remain a cross-paper reference with the limits recorded in the protocol
audit.

## Training record

The verified [training-completion record](training-completion.json) identifies
the final weights and confirms all Adam update counters at 20,000. The policy
has 73,423,367 trainable parameters. Its training loop took 2.93 hours on two
H100 PCIe GPUs, excluding feature extraction. The [exact config](training-config.json)
and [complete scalar log](training-metrics.jsonl) are available independently
of benchmark scores; training loss is not a task-success measurement.

The release supports two-GPU DDP with a global batch of eight episodes, four
per rank. Ragged episode losses are weighted by valid target counts across
ranks; action microbatch 128 reduces small-kernel overhead without changing
the objective. Both properties are covered by gradient-equivalence tests.

Data reuse preserves the recovered 861 training / 100 validation membership,
with 200 held-out initial states kept separate. Frozen agent-view DINOv3 and
CLIP features are reused. The setup uses 224×224 images, history stride 10 at
20Hz (2Hz observations), temporal RoPE, capacity 144 observations, a 16×7 native
OSC_POSE action chunk, constant 1e-4 AdamW, and package min/range normalization.
The record includes the exact config, manifest checksum, source revision,
scalar metrics, checkpoint checksum and all 200 rollout outcomes.

[`training-runtime.json`](training-runtime.json) records package versions read
from the running training container, its image content ID, and GPU/driver details.
The two H100 PCIe devices use BF16 autocast and FP32 model/optimizer state;
training matrix multiplication allows TF32. Execution flags are checked against
the frozen training source. This record is not a complete environment lockfile;
the visual-extraction runtime is documented separately in the cache audit.

Table4 of the LIBERO-Mem paper will be treated as published reference data
unless the audit establishes matching input, split, checkpoint and evaluation
settings. Recovering its code does not establish which settings produced each
published row. No strict superiority or matched-compute claim is assumed.

See [comparison-protocol.md](comparison-protocol.md) for the recovered source
audit and explicit comparison limits. The simulator's recorded upstream pin
is distinguished from its actual snapshot hashes. Fresh evaluation uses
independent recorded trial seeds and reports both task-mean and pooled
completion. Partial smoke runs cannot pass `scripts/verify_evaluation.py`.

[`evaluation-environment.json`](evaluation-environment.json) records the actual
simulator-venv packages, vendored source versions and EGL driver. A
[complete Python-source comparison](upstream-source-audit.json) confirms that
all 584 files, including vendored robosuite/robomimic, match the official fixed
commit. Native assets and a complete operating-system lock are outside that
source-comparison check.

## Dataset snapshots

The [data-source record](data-sources.json) identifies the two official dataset
repositories, fixed revisions and HDF5 LFS hashes. All ten raw files used for
initial states and horizons match the official byte hashes. RGB entries record
upstream expected hashes; they do not claim that the entire RGB corpus was
re-extracted during this run.

## Exact task conditioning

[`frozen-text.json`](frozen-text.json) contains the ten frozen float32 task vectors
used by this experiment, ordered T1–T10, as checksum-protected NPY bytes. Restore
them with `scripts/restore_text_features.py`; the root README has the command.
The original text-encoder revision was not recovered. The supplied vectors make
the numerical condition reproducible without claiming a recovered encoder pin.

## Frozen visual cache

[`backbone-source.json`](backbone-source.json) pins the timm DINOv3 weight file,
its official download revision and SHA-256. The existing local file matches
that hash and the launch manifest.

The launch reuses frozen DINO patches from the recovered cache. Preparation is
ordered by numeric task ID, not filename order, to retain the sampler's episode
indexing. The root command selects `--feature-layout legacy-paired32`: 32-time
blocks, GPU resizing, BF16 backbone evaluation, and two copies of each agent
image followed by discarding the duplicate. The policy input remains one camera.

[`feature-cache-audit.json`](feature-cache-audit.json) checks nine fixed
preprocessed views from T1, T3 and T9. Converted patches match the original
agent-view block byte for byte, and the released reproduction function re-encodes
those samples exactly in the recorded environment. This is not a re-extraction
of the complete raw RGB corpus. Changing extraction batch or numerical backend
can change feature values.

[`prepared-manifest.json`](prepared-manifest.json) is the exact launch manifest;
its SHA-256 is the value recorded in the checkpoint. It contains public episode
IDs and relative cache paths. Generic preparation can serialize different
metadata even when it produces the same splits and model inputs.
Use `scripts/materialize_reference_cache.py` after preparation, as shown in the
root README. It validates the semantic contract and creates a symlink view with
the exact launch manifest bytes. The source arrays remain required and must not
be changed. The view checks headers and task-text bytes, not numerical equality
of the complete visual corpus; its provenance file records this limitation.

[`prepared-sha256.txt`](prepared-sha256.txt) fingerprints every feature/action
array, the task-text matrix and the exact manifest read from the training cache.
[`cache-fingerprints.json`](cache-fingerprints.json) records the file sizes and
checksum-list hash. To compare a canonical cache, run from the repository root:

```bash
FPP_CACHE_CHECKSUMS="$(realpath results/no-eoe-20k/prepared-sha256.txt)"
(cd /datasets/fpp-prepared && sha256sum --quiet --check "$FPP_CACHE_CHECKSUMS")
```

Exit status zero with no output means all files match byte for byte. This reads
the complete cache. NPY serialization and extraction numerics can affect these
hashes; matching metadata alone is weaker than matching file contents. The list
identifies the cached inputs of this run, not a complete raw-RGB re-extraction.
[`checksum-crosscheck.json`](checksum-crosscheck.json) records an independent
system `sha256sum` check of seven entries and a file-size check of every entry.

## Export a completed reproduction

```bash
python -m scripts.export_run \
  --training outputs/no-eoe-20k \
  --evaluation outputs/eval-seed42 \
  --data-root /datasets/fpp-prepared \
  --output /tmp/fpp-public-results \
  --policy-output /tmp/fpp-no-eoe-20k.pt
```

The exporter requires a no-EOE 20K checkpoint, matching training/data/evaluation
hashes, final Adam counters, complete training metrics, and all 200 verified
trials. It derives `tasks.csv` and article-ready `results.json` from those same
records. It copies the raw observations selected by the evaluator and removes
local run paths from public JSON. Output destinations must be fresh.

The optional policy artifact omits optimizer state and keeps the inference
contract. Its SHA-256 is recorded in `evidence.json`. Large weights are stored
separately from the small files committed here. This command stages files only;
publication follows inspection of the actual completed run.

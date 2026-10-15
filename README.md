# FPP · Full-History Patch Policy

FPP retains spatial visual patches over a sampled observation history. A
block-causal Transformer contextualizes the current patches; a flow-matching
decoder reads that block and a task prefix. This release is restricted to
LIBERO-Mem simulation; private real-robot data and controllers are excluded.

[Web blog](https://ycma8.github.io/fpp-libero-mem/) · [LIBERO-Mem](https://libero-mem.github.io/)

## Results

FPP is evaluated on the ten LIBERO-Mem tasks using the **20K checkpoint**,
with 20 trials per task and 200 trials per policy.

| Policy | Whole-task success | Task-mean subgoal completion | Pooled subgoal completion |
|---|---:|---:|---:|
| FPP | **101/200 · 50.50%** | **68.55%** | **414/620 · 66.77%** |
| Two-frame Diffusion Policy | 5/200 · 2.50% | 17.99% | 110/620 · 17.74% |

Each method has 20 trials per task. Task mean weights the ten tasks equally;
pooling weights subgoal opportunities equally. FPP has 17 overshot trials. DP
receives two consecutive frames, providing short-term motion information.
Pretraining, batch and compute differ; this is not an isolated memory ablation.

The FPP checkpoint was selected after inspecting several test evaluations.
Results are exploratory and use one training seed. Published Table 4 values are
cross-paper references, not a matched-protocol comparison. See the
[reference record](results/libero-mem-reference/README.md) and
[comparison audit](results/no-eoe-20k/comparison-protocol.md). Verify all 400
original trial records with Python's standard library:

```bash
python scripts/verify_results.py
```

Additional experiments with different training recipes are preserved in the
[experiment archive](results/no-eoe-20k/README.md); their results are separate
from the table above.

## Install

Use Python 3.10+ and install PyTorch 2.5+ for your CUDA environment first:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

Long histories require CUDA FlexAttention. CPU is suitable for short smoke tests.

## Prepare reference inputs

Obtain the [RGB demonstrations](https://huggingface.co/datasets/libero-mem/LIBERO-Mem)
and the separate [raw initial-state export](https://huggingface.co/datasets/libero-mem/LIBERO-Mem-Raw).
The [source record](results/no-eoe-20k/data-sources.json) pins their revisions
and file hashes. Use the [pinned DINOv3 weights](results/no-eoe-20k/backbone-source.json).
The supplied frozen CLIP task vectors define the task condition and are
identified by artifact hashes:

```bash
python scripts/restore_text_features.py \
  --artifact results/no-eoe-20k/frozen-text.json \
  --output /datasets/fpp-task-text.npy
python scripts/prepare_cache.py \
  --raw-glob '/datasets/libero-mem-rgb/*.hdf5' \
  --text-features /datasets/fpp-task-text.npy \
  --text-encoder 'CLIP ViT-B/32 projected task vectors; supplied frozen bytes' \
  --dino-checkpoint /models/dinov3/model.safetensors \
  --output /datasets/fpp-reference \
  --history-stride 5 --max-observations 144 --normalization symmetric \
  --batch 64 --feature-layout legacy-paired32 \
  --reference-split results/libero-mem-reference/split-manifest.json
python scripts/check_prepared_data.py /datasets/fpp-reference
```

This split preserves 861 training / 100 validation episodes and 200 separate held-out
initial states. Test images and action targets do not enter training. Exact-zero
gripper labels map to open (-1). Native 7D OSC_POSE delta actions use training-only
midpoint/half-range normalization; constant dimensions restore their fixed value.

RGB is padded to square and resized to 224×224. The frozen encoder returns 196
agent-view patches. Legacy extraction duplicates agent images within 32-time BF16
batches, then discards the duplicate: the policy still sees one camera. The
[feature-cache audit](results/no-eoe-20k/feature-cache-audit.json) records sample
comparisons and extraction settings. Cache hashes identify a specific extraction
and metadata format; the original manifest and the converted single-camera cache
have separate provenance records.

## Train

```bash
python train.py --data-root /datasets/fpp-reference \
  --output outputs/fpp-reference-seed42 --steps 20000 --batch 8 --seed 42 \
  --temporal-position rope_spatial --lr-schedule reference \
  --adam-beta2 0.95 --action-microbatch 16 \
  --source-revision "$(git rev-parse HEAD)"
```

Defaults use temporal RoPE plus fixed 2D spatial positions, AdamW
betas (0.9, 0.95), and the measured LR curve. The curve uses 500 warmup steps and
a 30K budget, including continuation boundaries 7000/7086; a 20K endpoint does
not rescale it. Architecture: six temporal/eight action layers at width 512,
16 × 7 chunks, gradient clip 1.0 and weight decay 0.01.

Training also appends one valid final observation paired with the last action
repeated over a chunk (EOE). This is terminal supervision, not a stop token or
a hard termination rule. It is enabled by default; `--no-append-eoe` disables it.

The trainer applies this LR curve in one process. Continuation settings for the
reported run are documented in the [training record](results/libero-mem-reference/README.md).
Multi-GPU DDP is available through `torchrun`; `--batch` is global and rank losses
are weighted by valid targets. The reported model was trained on one RTX 4090 D.

## Evaluate

Install [LIBERO-Mem](https://github.com/libero-mem/libero-mem) at
`0eee0defa4024e51511b6e15a79f30e9620209de`. Standard LIBERO has different goal
counters. The inspected [runtime](results/no-eoe-20k/evaluation-environment.json)
uses MuJoCo 3.2.2, Gym 0.26.2 and vendored robosuite 1.4.1 / robomimic 0.3.1.

```bash
python evaluate.py \
  --checkpoint outputs/fpp-reference-seed42/step_020000.pt \
  --raw-glob '/datasets/libero-mem-raw/*.hdf5' \
  --data-root /datasets/fpp-reference \
  --bddl-root /path/to/libero-mem/libero/libero/bddl_files/libero_mem \
  --dino-checkpoint /models/dinov3/model.safetensors \
  --simulator-revision 0eee0defa4024e51511b6e15a79f30e9620209de \
  --image-transform flipud --output outputs/eval-reference --device cuda
python scripts/verify_evaluation.py --evaluation outputs/eval-reference \
  --manifest /datasets/fpp-reference/prepared/manifest.json
```

Use `--dry-run` without checkpoint/output for membership checks. Inspect reset
images against the RGB export: orientation must match. XML/state reset, 20 settling
steps, 5 executed controls, full task horizons and overshoot checks are explicit.
Whole-task success requires all subgoals and the final predicate without overshoot.
The public evaluator independently seeds each trial. Evaluator sources and the
protocol for the reported results are recorded in the reference directory.

## Web blog

The [web blog](https://ycma8.github.io/fpp-libero-mem/) describes the method and results. It uses
research-article-template and gives author attribution in its footer citation.
The blog is the only publication format maintained in this repository.

With Node 20 or newer:

```bash
cd article
npm ci
npm run build
npm run preview
```

Weights, dense caches, datasets and simulator assets are separate dependencies.
Cite FPP, LIBERO-Mem and the pretrained backbones; third-party assets retain
their licenses.

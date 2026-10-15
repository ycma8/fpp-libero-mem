---
tags:
  - research-article-template
---

# FPP web blog

This is the web blog for the simulation-only FPP release. Author attribution
appears in the footer citation; the top of the article has no author bar.
There is no separate PDF or project-page edition in this repository.

[Read the blog](https://ycma8.github.io/fpp-libero-mem/).

The article reports 101/200 whole-task successes, 68.55% task-mean subgoal
completion and 414/620 pooled completion on LIBERO-Mem. Training settings and
per-trial results are provided in `../results/libero-mem-reference/`.

With Node 20 or newer, from this directory:

```bash
npm ci
npm run build
npm run preview
```

The preview serves the article at `http://localhost:8080/`. GitHub Pages builds
use `SITE_URL=https://ycma8.github.io` and `BASE_PATH=/fpp-libero-mem` so assets
load from the repository subdirectory. The Pages workflow builds and publishes
the blog when `main` is updated.

Edit `src/content/article.mdx`; figures are in `src/content/embeds/`.
The authoritative measurement record is `../results/libero-mem-reference/`.

The task-results figure reads `src/content/assets/data/experiment-status.json`.
This file is derived from the verified reference summary. The figure checks
per-task and aggregate counts before drawing bars; this display check does not
replace the raw-trial verifier. For a new experiment, verify and export its raw
records before replacing that JSON and updating the surrounding prose.

Adapted from [research-article-template](https://huggingface.co/spaces/tfrere/research-article-template),
revision `c75fc29c361755f66c46ae6e83d958b10c4d5feb`, by Thibaud Frere.
Its CC BY 4.0 attribution and license are retained in `LICENSE.template`.
The model code's license is independent of that template license.

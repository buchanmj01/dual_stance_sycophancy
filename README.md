# Dual-Stance Evaluation of Sycophancy

**Paper:** [Dual-Stance Evaluation of Sycophancy: The Structure of Agreement and the Limits of Intervention](ARXIV_LINK)

Standard activation steering for sycophancy passes single-stance evaluation - but does it actually target sycophancy, or does it just suppress agreement? This repo contains the full codebase for a dual-stance evaluation method that distinguishes the two. Applied to centroid-difference steering on Llama-3-8B-Instruct, we find:

1. **The steering direction is non-specific** - it reduces agreement with factually correct statements as well as sycophantic ones.
2. **The effects are highly structured** — dual-stance consistency predicts steering susceptibility (*r* = 0.88 in-sample, *r* = 0.84 out-of-sample on 12 novel topics).
3. **The geometry is puzzling** — sycophantic and factual agreement occupy distinct activation subspaces, yet the steering direction projects equally onto both.

## Repository contents

```
├── dual_stance_sycophancy.ipynb   # Full codebase (Colab notebook)
├── figures/                       # Generated figures (PDF + PNG)
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

1. Open the notebook in Google Colab or run it locally. Select a GPU runtime (Colab T4 is sufficient).
2. You will need a Hugging Face token with access to `meta-llama/Meta-Llama-3-8B-Instruct`.
3. Run cells sequentially from Cell 1. Total runtime is approximately 8–12 hours on a T4 for the full pipeline.

**Tip:** The analysis and figure-generation cells (7–10, 16–19) do not require a GPU and run in under a minute each, once the earlier cells have saved their outputs.

**Tip:** If you only want to explore the analysis and figures, run the GPU-dependent cells first (1–6), then the analysis cells (7–10, 16–19) can be run independently. The notebook saves intermediate results to disk between stages.

## Requirements

The notebook installs its own dependencies. The main packages are:

- `transformers`, `accelerate`, `bitsandbytes` (model loading, 4-bit quantisation)
- `scikit-learn` (linear probes)
- `numpy`, `matplotlib` (analysis and figures)
- A Hugging Face account with Llama 3 access (for full reproduction only)

## Citation

```TBC
```

## Contact

Matthew James Buchan — [buchanmj01@gmail.com](mailto:buchanmj01@gmail.com)

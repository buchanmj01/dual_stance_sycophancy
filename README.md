# Dual-Stance Evaluation of Sycophancy

**Paper:** [Dual-Stance Evaluation of Sycophancy: The Structure of Agreement and the Limits of Intervention](ARXIV_LINK)

Standard activation steering for sycophancy passes single-stance evaluation — but does it actually target sycophancy, or does it just suppress agreement? This repo contains the full codebase for a dual-stance evaluation method that distinguishes the two. Applied to centroid-difference steering on Llama-3-8B-Instruct, we find:

1. **The steering direction is non-specific** — it reduces agreement with factually correct statements as well as sycophantic ones.
2. **The effects are highly structured** — dual-stance consistency predicts steering susceptibility (*r* = 0.88 in-sample, *r* = 0.84 out-of-sample on 12 novel topics).
3. **The geometry is puzzling** — sycophantic and factual agreement occupy distinct activation subspaces, yet the steering direction projects equally onto both.

## Repository contents

```
├── dual_stance_sycophancy.ipynb   # Full codebase (Colab notebook)
├── requirements.txt
├── LICENSE
└── README.md
```

## Notebook structure

The notebook is organised into sequential cells. GPU-dependent cells (marked with runtime estimates) must be run on Colab or a CUDA-capable machine. Analysis and figure cells load saved results from disk and can be run independently.

| Cell | Description | GPU | Runtime (T4) |
|------|-------------|-----|-------------|
| 1 | Setup, model loading, items, prompts, hooks | Yes | ~5 min |
| 2 | Collect activations (train set) | Yes | ~60 min |
| 3 | Probe training, held-out evaluation, per-category detection | Yes | ~60 min |
| 4a | *Bridge cell* — reload state after disconnect (for Cell 5a) | No | <1 min |
| 5a | Causal steering test, per-category analysis, sycophancy validation | Yes | ~90 min |
| 4b | *Bridge cell* — reload state + random direction (for Cell 5b) | No | <1 min |
| 5b | Random-direction control experiment | Yes | ~60 min |
| 6 | α-ablation with checkpointing | Yes | ~120 min |
| — | Headroom analysis (matched baselines) | No | <1 min |
| — | Continuous dissociation analysis | No | <1 min |
| — | Subspace analysis | No | <1 min |
| — | Out-of-sample steering prediction | Yes | ~120 min |
| — | Multi-layer steering diagnostic | Yes | ~120 min |
| — | Cross-layer representation stability | No | <1 min |
| — | Prompt variation diagnostic | Yes | ~120 min |
| — | Mistral-7B diagnostic | Yes | ~120 min |
| — | Mistral higher α diagnostic | Yes | ~15 min |
| — | All figures (figs 0–6, appendix A1–A2) | No | <1 min |

**Bridge cells (4a, 4b)** are optional — use them to restore state after a Colab disconnect without re-running the GPU-heavy cells. Run Cell 1 first, then the appropriate bridge cell.

**Total runtime** for the full pipeline is approximately 15–18 hours on a T4 GPU. The Mistral cells require loading a second model and should be run in a separate session after freeing the Llama model from GPU memory.

## Quick start

1. Open the notebook in Google Colab with a T4 GPU runtime.
2. Set your Hugging Face token as a Colab secret named `HF_TOKEN`. You need access to `meta-llama/Meta-Llama-3-8B-Instruct` and `mistralai/Mistral-7B-Instruct-v0.3`.
3. Run cells sequentially from Cell 1. Intermediate results are saved to Google Drive at `/content/drive/MyDrive/sycophancy_v3/`.

**If you only want to explore the analysis and figures:** run Cell 1 (for imports and item definitions), then skip to the analysis and figure cells. These load saved results from disk and do not require a GPU.

## Requirements

The notebook installs its own dependencies via `!pip install`. The main packages are:

- `transformers`, `accelerate`, `bitsandbytes`, `huggingface_hub` — model loading, 4-bit quantisation
- `scikit-learn` — linear probes, PCA
- `scipy` — rank correlations, statistical tests
- `numpy`, `matplotlib` — analysis and figures
- A Hugging Face account with Llama 3 and Mistral access (for full reproduction)

A `requirements.txt` is provided for local development, though the notebook is designed to run on Google Colab.

## Saved outputs

The notebook saves intermediate results to Google Drive, enabling cells to be run independently:

- `activations_train.pt` — cached activations from Cell 2
- `probe_results.json` — probe training results from Cell 3
- `steering_results.json` — baseline and steered responses from Cell 5a
- `random_control_results.json` — random direction control from Cell 5b
- `ablation_checkpoint.json` — α-ablation results from Cell 6 (resumable)
- `oos_steering.json` — out-of-sample prediction results
- `multi_layer_diagnostic.json` — multi-layer steering results
- `prompt_variation_baseline.json`, `prompt_variation_steering.json` — prompt variation results
- `mistral_diagnostic.json` — Mistral replication results
- `fig*.pdf`, `fig*.png` — all figures

## Citation

```
TBC
```

## Contact

Matthew James Buchan — [buchanmj01@gmail.com](mailto:buchanmj01@gmail.com)
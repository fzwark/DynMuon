# DynMuon: A Dynamic Spectral Shaping View of Muon

![LLM Training](https://img.shields.io/badge/LLM_Training-yellow.svg) ![Dynamic Spectral Shaping](https://img.shields.io/badge/Dynamic_Spectral_Shaping-blue.svg) ![Muon](https://img.shields.io/badge/Muon-green.svg)

## Introduction

In recent years, Muon has emerged as the dominant method for training large language models, and transformers more broadly. The essential difference, when compared to standard gradient descent methods, is to replace the usual update matrix $M=U\Sigma V^\top$ with its polar factor $UV^\top$.
In this work, we consider a class of Muon-like updates, where we replace the update $M$ with $U\Sigma^p V^\top$ for some parameter $p$. 
We call this a "spectral-shaping" operation, and develop a theory of how to pick $p$ which depends on (a) local curvature of the loss function, (b) noise stemming from stochastic gradients and label noise, and (c) training stage.
Our theory and experimentation reveal a previously overlooked behavior: positive $p$ helps early by emphasizing high-curvature directions and accelerating signal contraction, while mildly negative $p$ helps later by reallocating update strength toward low-curvature directions that still contain useful training signals.
Building on the insight, we propose DynMuon, an efficient dynamic spectral shaping method that schedules $p$ from positive to mildly negative over training.
Extensive experiments across model sizes, architectures, and training settings show that DynMuon consistently achieves lower validation loss than Muon, while requiring **10.6-26.5%** fewer steps to reach the same target loss.

## Updates

- [2026.05.31] 🚀 Our code is now released! DynMuon achieves a validated [Track 3 result of 3175 steps](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization), placing it among the strongest **standalone Muon-style optimization methods** on the modded-nanoGPT optimization benchmark.

## Setup

Install the package and training dependencies:

```bash
cd DynMuon
pip install -e .[train]
```


Download pretokenized FineWeb dataset (10B):

```bash
python data/cached_fineweb100B.py 100
```

The argument `100` downloads the first 100 training shards. 

## Run Training

To train a 127M parameter (512/24/8) GPT-style model with DynMuon, run:

```bash
torchrun --standalone --nproc_per_node=1 train_gpt.py \
    --optimizer dynmuon \
    --scalar_opt adamw \
    --lr 0.01 \
    --batch_size 512 \
    --device_batch_size 64 \
    --sequence_length 1024 \
    --num_iterations 20000 \
    --model_dim 512 \
    --n_layer 24 \
    --n_head 8 \
    --dynmuon_pmax 1.0 \
    --dynmuon_pmin -0.25 \
    --dynmuon_w 0.04 \
    --dynmuon_tau 0.04
```

**Key parameters:**
- `model_dim=512` / `n_layer=24` / `n_head=8`: Model architecture (512 hidden dim, 24 layers, 8 heads)
- `batch_size=512` / `device_batch_size=64`: Global and per-device batch sizes
- `sequence_length=1024`: Context length for training
- `num_iterations=20000`: Total training steps
- `lr=0.01`: Learning rate for the main optimizer
- `dynmuon_pmax=1.0`, `dynmuon_pmin=-0.25`: Spectral schedule range (transitions from positive to mildly negative)
- `dynmuon_w=0.04`, `dynmuon_tau=0.04`: Logistic schedule width and center ratios
- `optimizer` Optimizer choice. Common choices include `dynmuon`, `muon`, `normuon`, and `adamw`.



## Citation

If you use this codebase, please consider citing our paper:

```bibtex
@misc{wu2026dynmuondynamicspectralshaping,
      title={DynMuon: A Dynamic Spectral Shaping View of Muon}, 
      author={Fangzhou Wu and Rikhav Shah and Sandeep Silwal and Qiuyi Zhang},
      year={2026},
      eprint={2605.17109},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.17109}, 
}
```

## Acknowledgements

This codebase builds on the [Dion optimizer codebase](https://github.com/microsoft/dion). We thank the Dion team for open-sourcing their implementation.
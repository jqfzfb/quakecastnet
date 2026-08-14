# QuakeCastNet

**Probabilistic Multi-Horizon Spatiotemporal Forecasting of Injection-Induced Seismicity in Geothermal Systems**

Companion code for the manuscript by Zhengfa Bi (Lawrence Berkeley National Laboratory) and Nori Nakata (Lawrence Berkeley National Laboratory; MIT).

## Overview

Fluid injection and reservoir stimulation at geothermal fields can trigger induced seismicity, and reliable forecasting is complicated by the coupled interactions between the solid Earth and the injected fluids. **QuakeCastNet** is an interpretable deep learning framework that issues probabilistic, multi-horizon forecasts of seismicity rates and their spatial distribution, combining historical seismicity with past and scheduled injection operational data.

The framework consists of three components:

- **Temporal branch** — an extended Temporal Fusion Transformer (TFT) produces quantile forecasts of the seismicity rate, with multi-seed ensemble inference for stability.
- **Post-injection decay adaptation** — an empirical decay shape learned from historical shut-in episodes adapts the forecast once injection stops.
- **Spatial branch** — a regime-adaptive spatial density model (RA-SDM) blends a recency-weighted kernel estimate, a physical spreading kernel, and local triggering around larger historical events into a normalized density field, with the injection state selecting between stable and active regimes.

Case studies at two geothermal fields, **The Geysers** (California) and **Utah FORGE** (2024 stimulation), reveal site-specific seismic responses governed by hydromechanical conditions: the model infers a longer predictive memory at The Geysers and a more immediate relation to operations at Utah FORGE. Attention weights and input-variable ablations provide interpretability, and temporal and spatial forecasts are benchmarked against statistical reference models.

## Repository structure

```
quakecastnet/
├── utahforge2024_train.ipynb      # Utah FORGE 2024 case: data → training → evaluation → spatial forecast
├── geysers929_train.ipynb         # The Geysers case: same end-to-end pipeline
├── draw1.py, draw2.py             # scripts reproducing the figures of the paper
├── utils.py                       # data processing and evaluation utilities
├── models/
│   ├── branchwise_transport_spatial.py   # QuakeCastNet model used by both notebooks
│   ├── tft_model_new.py                  # extended TFT with composite quantile loss
│   ├── shutin_decay_tft.py               # post-injection (shut-in) decay modeling
│   ├── gated_rolling.py                  # rolling forecast utilities
│   ├── spatial_branch_eval.py            # spatial branch evaluation
│   └── verify_shutin_decay.py, verify_shutin_bias.py   # checks of the shut-in decay assumption
├── preparation/
│   ├── dataset/                   # ready-to-use model input tensors (.npy) for both sites
│   └── raw_data/                  # processed seismicity catalogs and injection records
├── pytorch_forecasting/           # vendored, *modified* copy of pytorch-forecasting (see below)
├── requirements.txt
└── LICENSE
```

### Note on the vendored `pytorch_forecasting/`

This repository ships a **modified** copy of [pytorch-forecasting](https://github.com/sktime/pytorch-forecasting) (upstream development snapshot reporting version `1.1.1`, which is not available on PyPI; MIT License, Copyright 2020 Jan Beitner — see `pytorch_forecasting/LICENSE`). The copy contains local modifications, including experimental graph-attention submodules in `models/temporal_fusion_transformer/sub_modules.py` that are **not** used by the final models. Shipping the vendored copy guarantees the exact code paths used for the paper; it takes precedence automatically when running from the repository root.

## Installation

```bash
git clone https://github.com/<your-github-username>/quakecastnet.git
cd quakecastnet
conda create -n quakecastnet python=3.13
conda activate quakecastnet
pip install -r requirements.txt
```

The reference environment used for the paper is Python 3.13 with PyTorch 2.11.0 (CUDA 12.6 build). For a CUDA-enabled PyTorch, install the wheel matching your platform from [pytorch.org](https://pytorch.org) instead of the plain `torch==2.11.0` pin.

## Data

The `preparation/` directory contains everything needed to run the notebooks:

- `preparation/dataset/` — model-ready input tensors: `UtahForge2024/` (~10 MB), `Geysers929/` (~19 MB), `UtahForge2022/` (~0.2 MB, supplementary).
- `preparation/raw_data/` — processed seismicity catalogs (CSV) and well/operation records underlying the tensors.

Both datasets are derived from public data for the two field sites; see the *Study Sites and Forecasting Configuration* section of the manuscript for provenance and processing details.

## Reproducing the results

Each case study has a self-contained notebook with the same pipeline (load data → build dataset and dataloaders → train the TFT → evaluate performance → spatial distribution of seismicity → single-sample visualization):

| Notebook | Field site |
|---|---|
| `utahforge2024_train.ipynb` | Utah FORGE (April 2024 stimulation) |
| `geysers929_train.ipynb` | The Geysers (well 929 area) |

Training runs write Lightning logs and checkpoints to `logs/` (three random seeds per site, following the multi-seed ensemble inference described in the paper). This directory is gitignored because of its size and is regenerated by rerunning the notebooks. `draw1.py` and `draw2.py` produce the figures of the paper from the trained-model outputs.

## Citation

If you use this code, please cite the accompanying manuscript:

> Bi, Z. & Nakata, N. *Probabilistic Multi-Horizon Spatiotemporal Forecasting of Injection-Induced Seismicity in Geothermal Systems*. Manuscript under review.

Citation information (journal, DOI) will be added upon publication.

## License

The code in this repository is released under the [MIT License](LICENSE), except the `pytorch_forecasting/` directory, which is distributed under its own MIT License (Copyright 2020 Jan Beitner; see `pytorch_forecasting/LICENSE`).

## Contact

Zhengfa Bi — [zfbi@lbl.gov](mailto:zfbi@lbl.gov)

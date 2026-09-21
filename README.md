# QuakeCastNet

Code accompanying *Probabilistic Multi-Horizon Spatiotemporal Forecasting of Injection-Induced Seismicity in Geothermal Systems* by Zhengfa Bi (Lawrence Berkeley National Laboratory) and Nori Nakata (Lawrence Berkeley National Laboratory; MIT).

## Overview

QuakeCastNet forecasts seismicity rates and spatial probability distributions at geothermal sites. The inputs include historical seismicity, injection rates, and operating pressure. The temporal model can also use known future injection schedules.

The framework contains three components:

- An extended Temporal Fusion Transformer (TFT) produces quantile forecasts of the seismicity rate. Predictions from multiple checkpoints are combined into an ensemble.
- A shut-in correction uses decay patterns from historical injection pauses to adjust the Utah FORGE forecasts after injection stops.
- A spatial model combines a kernel density estimate weighted by event age with a distance-based background. An activity and injection gate and a magnitude-trigger term are enabled for Utah FORGE and disabled in the default Geysers configuration.

The notebooks cover The Geysers well 929 area and the April 2024 Utah FORGE stimulation. They include temporal model comparisons, spatial evaluation, and individual forecast visualizations.

## Repository structure

```text
.
├── geysers929_train.ipynb
├── utahforge2024_train.ipynb
├── draw1.py
├── draw2.py
├── utils.py
├── models/
│   ├── tft_model_new.py
│   ├── gated_rolling.py
│   ├── shutin_decay_tft.py
│   ├── branchwise_transport_spatial.py
│   ├── spatial_branch_eval.py
│   ├── spatial_gate_ablation.py
│   ├── etas.py
│   ├── etas_mancini.py
│   └── etas_kim_convolution.py
├── preparation/
│   ├── dataset/
│   └── raw_data/
├── pytorch_forecasting/
├── tests/
│   └── test_spatial_gate_ablation.py
├── requirements.txt
└── LICENSE
```

`draw1.py` and `draw2.py` provide plotting functions called by the notebooks. `utils.py` provides data processing, baseline forecasting, and evaluation functions. The ETAS implementations are included in `models/`, so the comparisons do not require a separate parent project.

`models/spatial_gate_ablation.py` provides an additional comparison of the baseline, gate only, magnitude trigger only, and both mechanisms together. Its tests check time boundaries and compatibility with the original spatial model. The current main notebooks do not call this auxiliary module.

## Installation

The local validation environment uses Python 3.10.20 and PyTorch 2.11.0 with CUDA 12.6. The training cells request one CUDA GPU. Use a PyTorch build compatible with your platform while keeping the version aligned with `requirements.txt`.

Download the project and run the following commands from the directory containing this README:

```bash
conda create -n quakecastnet python=3.10
conda activate quakecastnet
python -m pip install -r requirements.txt
python -m pip install pytorch-optimizer==3.10.1
python -m ipykernel install --user --name quakecastnet --display-name "QuakeCastNet"
jupyter notebook
```

Select the **QuakeCastNet** kernel when opening a notebook. The additional `pytorch-optimizer` package is required by the notebooks' `optimizer="ranger"` setting and is currently installed separately from `requirements.txt`.

The repository contains a modified copy of [pytorch-forecasting](https://github.com/sktime/pytorch-forecasting). Run the notebooks from the project root so Python imports that copy. Its local version string is `1.1.1`. Installing a different copy from PyPI can change model behavior and checkpoint compatibility. The bundled TFT submodules import `torch-geometric`, which is included in the requirements even though the graph-attention path is not used by the current models.

To check which copy Python imports, run:

```bash
python -c "import pytorch_forecasting; print(pytorch_forecasting.__file__)"
```

The printed path should point to this repository's `pytorch_forecasting/` directory.

## Data and forecast settings

The processed inputs are included under `preparation/dataset/`:

- `Geysers929/seismic_inject_dataset929_ext.npy` contains the Geysers inputs.
- `UtahForge2024/seismic_inject_dataset2024_ext.npy` contains the Utah FORGE 2024 inputs.
- `UtahForge2022/seismic_inject_dataset2022.npy` supplies historical injection episodes for the Utah shut-in correction.

`preparation/raw_data/` contains the processed seismicity catalogs and Utah well records used for spatial evaluation. See the accompanying manuscript for data sources and processing details.

Both notebooks use 200 encoder steps and 80 forecast steps after downsampling.

| Setting | The Geysers | Utah FORGE |
|---|---|---|
| Notebook | [geysers929_train.ipynb](geysers929_train.ipynb) | [utahforge2024_train.ipynb](utahforge2024_train.ipynb) |
| Time step | 4 days | 8 minutes |
| Forecast horizon | 320 days | 10 hours 40 minutes |
| Rate and RMSE units | events/day | events/min |
| Forecast windows in the current data | 208 | 719 |

## Running the notebooks

The data files are included, but model checkpoints and prediction caches are excluded by `.gitignore` through the `logs/` and `*.ckpt` rules. A source download therefore requires either training or separately supplied checkpoints. The full method comparison also requires a separate forecast cache for the experiment without future injection inputs, as described below.

1. Run the import, configuration, data-loading, and dataloader cells in order. Review `NM_SEEDS`, which selects checkpoint directories, and the forecast settings for the chosen site.
2. Train with `DO_TRAINING=True`, or provide matching checkpoints under `logs/<log_name>/seed_<seed>/ckpts/`. The default is `DO_TRAINING=False`. Prediction uses the latest stored epoch for each requested seed directory.
3. Before running `Evaluate Model Performance`, ensure that a numeric results version directory exists. The results-path cell selects the highest existing `version_<number>` directory. For a fresh run with the default settings, create the directories shown below.
4. Run `Evaluate Model Performance` to generate or load TFT predictions. Run `Forecast Skill Comparison` once its required caches are available. The following sections evaluate the spatial forecasts and display individual samples.

Create the default results directories from the project root:

```bash
python -c "from pathlib import Path; Path('logs/geysers929_i200o80t2012-07-01/version_0').mkdir(parents=True, exist_ok=True); Path('logs/utahforge2024_i200o80t2024-04-06-06-06/version_0').mkdir(parents=True, exist_ok=True)"
```

For evaluation from existing caches, run the setup, data-loading, and results-path cells to restore the required variables. Training and TFT inference can then be skipped if all forecast caches listed below are present and match the loaded data.

### Forecast skill comparison

The Geysers comparison contains seven methods. Utah FORGE adds the forecast without the shut-in correction, giving eight methods.

| Method | Geysers cache | Utah FORGE cache |
|---|---|---|
| Proposed | `tft_results.npy` | `tft_shutin_results.npy` |
| Proposed (no decay) | Not included | `tft_results.npy` |
| Proposed (no fut. inj.) | `tft_nofuture_results.npy` | `tft_nofuture_results.npy` |
| GBDT | `gbdt_pz_results.npy` | `gbdt_pz_results.npy` |
| ETAS (rolling) | `etas_pz_results.npy` | `etas_pz_results.npy` |
| ETAS (global) | `etas_global_pz_results.npy` | `etas_global_pz_results.npy` |
| ETAS (inj.-driven) | `etas_mancini_pz_results.npy` | `etas_mancini_pz_results.npy` |
| InjConv | `etas_kim_pz_results.npy` | `etas_kim_pz_results.npy` |

Place the caches in the site's selected `logs/<log_name>/version_<number>/` directory. The notebooks generate the main TFT caches from checkpoints and fit the five statistical baselines when their caches are missing. Baseline parameters are fitted using data before `zero_date` and held fixed during forecasting.

Each window is evaluated using RMSE, MAE, Pearson correlation, cosine similarity, and SMAPE. For TFT outputs, the point prediction used for these metrics is the arithmetic mean across predicted quantiles. The combined score scales each metric across the compared methods within each window, then averages the five scaled values. Higher combined scores are better. 

## Spatial forecast of seismicity

The regime-adaptive spatial density model (RA-SDM) estimates where seismicity is likely to occur within each forecast window. It uses the earthquake catalog available at the forecast origin to combine a kernel density estimate weighted by event age with a spatial background. 

The spatial branch produces a probability map over the study grid. The temporal branch supplies the expected total number of events over the forecast window. Multiplying that total by each cell's probability gives the expected event count in that cell.

After preparing the data and temporal predictions, run the **Spatial Distribution of Seismicity** section in either notebook. The spatial cells use the forecast stored in `results_sets["current"]`. The resulting maps show expected event counts, observed earthquake locations, and contours enclosing regions of high forecast probability, alongside seismicity-rate and injection time series.

## Citation

If you use this code, please cite the accompanying manuscript:

> Bi, Z. and Nakata, N. *Probabilistic Multi-Horizon Spatiotemporal Forecasting of Injection-Induced Seismicity in Geothermal Systems*.

## License

The project code is distributed under the [MIT License](LICENSE). The bundled `pytorch_forecasting/` code has its own [MIT License](pytorch_forecasting/LICENSE), copyright 2020 Jan Beitner.

## Contact

Zhengfa Bi: [zfbi@lbl.gov](mailto:zfbi@lbl.gov).

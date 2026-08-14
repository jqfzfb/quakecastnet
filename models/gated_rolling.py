from __future__ import annotations

import dataclasses

import numpy as np

from .tft_model_new import RollingEnsembleForecaster


class GatedRollingEnsembleForecaster(RollingEnsembleForecaster):

    @staticmethod
    def _gate(pool, n_train) -> bool:
        return len(pool) >= n_train


def recenter_quantile_bands(point_source, band_source, atol=1e-9):
    assert len(point_source) == len(band_source), "两侧窗口数不一致"
    out = []
    for i, (rp, rb) in enumerate(zip(point_source, band_source)):
        tp = np.asarray(rp["target"], float)
        tb = np.asarray(rb["target"], float)
        assert np.array_equal(tp, tb), f"窗口 {i} target 不一致 — 两侧窗口未对齐"
        qp = np.asarray(rp["prediction"], float)
        qb = np.asarray(rb["prediction"], float)
        assert qp.ndim == 2 and qp.shape == qb.shape, f"窗口 {i} 预测形状不一致"
        shift = qp.mean(axis=1) - qb.mean(axis=1)
        q_new = qb + shift[:, None]
        assert np.allclose(q_new.mean(axis=1), qp.mean(axis=1), atol=atol), \
            f"窗口 {i} 均值保持断言失败"
        rd = dict(rp)
        rd["prediction"] = q_new
        out.append(rd)
    return out


def run_gated_recentered(config, *, ckpt_paths, training_template,
                         train_data_list, apply_data_list, target,
                         known_reals, time_axis, norm_stats):
    fc = GatedRollingEnsembleForecaster(config)
    run_kwargs = dict(
        ckpt_paths=ckpt_paths, training_template=training_template,
        train_data_list=train_data_list, apply_data_list=apply_data_list,
        target=target, known_reals=known_reals,
        time_axis=time_axis, norm_stats=norm_stats)
    results_rolling = fc.run(**run_kwargs)

    cfg_ens = dataclasses.replace(config, refit_every=10**9, ft_steps=1)
    fc_ens = RollingEnsembleForecaster(cfg_ens)
    results_ens = fc_ens.run(**run_kwargs)
    assert not any(e["finetuned"] for e in fc_ens.info_["refit_log"]), \
        "带源那遍发生了微调 — 纯集成前提被破坏"

    return recenter_quantile_bands(results_rolling, results_ens), results_ens, fc.info_

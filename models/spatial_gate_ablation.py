from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .branchwise_transport_spatial import (
    BranchwiseTransportSpatialModel,
    _median_mad_percentile,
    _rolling_stats,
)
from .spatial_branch_eval import (
    events_to_cells, js_divergence, kl_divergence, spatial_log_likelihood,
)


ARM_LABELS = {
    "baseline": "Baseline",
    "gate": "Gate only",
    "trigger": "Magnitude trigger only",
    "gate_trigger": "Gate and magnitude trigger",
}


class DailyGateSpatialModel(BranchwiseTransportSpatialModel):

    def __init__(self, sw_grid, *, config, artifact=None):
        super().__init__(sw_grid, config=config, artifact=artifact)
        if artifact is not None:
            self.RECENT_HOURS = float(artifact["rolling"]["window_hours"])


def build_daily_gate_artifact(catalog, injection_times, injection_values, *,
                              calibration_end, recent_days, stride_days):
    cutoff = pd.Timestamp(calibration_end).to_datetime64()
    tt = np.asarray(catalog["times"], dtype="datetime64[ns]")
    keep = tt <= cutoff
    it = np.asarray(injection_times, dtype="datetime64[ns]")
    ix = np.asarray(injection_values, dtype=float).reshape(-1)
    if it.shape != ix.shape or not np.all(np.isfinite(ix[it <= cutoff])):
        raise ValueError("Injection times and values must have the same length, with finite values throughout the calibration period.")
    available = (it <= cutoff) & np.isfinite(ix)
    it, ix = it[available], ix[available]
    if keep.sum() < 2 or not len(it) or not np.any(ix > 0):
        raise ValueError("The gate calibration period requires historical events and positive injection values.")
    la, lo = np.asarray(catalog["lats"])[keep], np.asarray(catalog["lons"])[keep]
    lat0, lon0 = float(np.median(la)), float(np.median(lo))
    first = pd.Timestamp(max(tt[keep].min(), it.min())).ceil("D")
    first += pd.Timedelta(days=2 * recent_days)
    endpoints = pd.date_range(first, pd.Timestamp(cutoff),
                              freq=pd.Timedelta(days=stride_days)).to_numpy()
    if len(endpoints) < 20:
        raise ValueError("The gate calibration period has fewer than 20 rolling endpoints; extend the calibration period.")
    counts, shifts = _rolling_stats(
        la, lo, tt[keep], endpoints, 24 * recent_days, lat0, lon0)
    positive = np.sort(ix[ix > 0])
    stride_minutes = stride_days * 24 * 60
    if stride_minutes <= 0 or stride_minutes != int(stride_minutes):
        raise ValueError("The gate stride must be a positive integer number of minutes.")
    return {
        "schema": "utahforge-branchwise-transport-v1",
        "purpose": "geysers-gate-ablation",
        "seismic_cutoff": str(pd.Timestamp(cutoff)),
        "projection": {"lat0": lat0, "lon0": lon0},
        "rolling": {
            "window_hours": float(24 * recent_days),
            "stride_minutes": int(stride_minutes),
            "first_endpoint": str(first),
            "n_reference_endpoints": len(endpoints),
            "n_finite_shifts": int(np.isfinite(shifts).sum()),
        },
        "gate": {
            "count_threshold": _median_mad_percentile(counts),
            "shift_threshold": _median_mad_percentile(shifts),
        },
        "injection": {
            "threshold": _median_mad_percentile(positive),
            "positive_reference": positive.tolist(),
        },
    }


def _predict(model, catalog, issue, injection_times, injection_values):
    return model.predict(
        catalog["lats"], catalog["lons"], catalog["mags"], catalog["times"], issue,
        injection_times=injection_times, injection_values=injection_values,
        return_diagnostics=True,
    )


def _cells_in_window(catalog, grid, issue, end):
    tt = np.asarray(catalog["times"], dtype="datetime64[ns]")
    mask = (tt > pd.Timestamp(issue).to_datetime64()) & (
        tt <= pd.Timestamp(end).to_datetime64())
    cells = events_to_cells(
        np.asarray(catalog["lats"])[mask], np.asarray(catalog["lons"])[mask],
        grid["lat_edges"], grid["lon_edges"], int(grid["n_grid"]))
    return cells[cells >= 0]


def tune_gate(catalog, grid, baseline_config, injection_times, injection_values,
              *, calibration_end, freeze_time, horizon_days, stride_days,
              recent_days_candidates, active_tau_candidates, progress=print):
    if baseline_config.beta != 0 or baseline_config.gamma != 1:
        raise ValueError("This experiment requires baseline beta=0 and gamma=1.")
    if baseline_config.variant != "unsharpened":
        raise ValueError("This experiment supports only the unsharpened branch.")
    freeze = pd.Timestamp(freeze_time)
    horizon = pd.Timedelta(days=horizon_days)
    origins = pd.date_range(
        pd.Timestamp(calibration_end), freeze - horizon - pd.Timedelta(1, "ns"),
        freq=pd.Timedelta(days=stride_days))
    if len(origins) < 10:
        raise ValueError("Fewer than 10 complete tuning windows precede the application period.")
    keep = np.asarray(catalog["times"], dtype="datetime64[ns]") < freeze.to_datetime64()
    cat = {key: np.asarray(value)[keep] for key, value in catalog.items()}
    it = np.asarray(injection_times, dtype="datetime64[ns]")
    ix = np.asarray(injection_values, dtype=float).reshape(-1)
    keep_inj = it < freeze.to_datetime64()
    it, ix = it[keep_inj], ix[keep_inj]
    cells = [_cells_in_window(cat, grid, t, t + horizon) for t in origins]
    n_cells = int(grid["n_grid"]) ** 2
    baseline = DailyGateSpatialModel(grid, config=baseline_config)
    base_gain = []
    for t, observed in zip(origins, cells):
        p, _ = _predict(baseline, cat, t, it, ix)
        ll, count = spatial_log_likelihood(p, observed)
        base_gain.append(ll / count + np.log(n_cells) if count else np.nan)
    base_gain = np.asarray(base_gain)
    if not np.isfinite(base_gain).any():
        raise ValueError("No tuning window contains observed events within the grid.")
    rows, artifacts = [], {}
    for recent_days in recent_days_candidates:
        artifact = build_daily_gate_artifact(
            cat, it, ix, calibration_end=calibration_end,
            recent_days=recent_days, stride_days=stride_days)
        artifacts[float(recent_days)] = artifact
        for tau in active_tau_candidates:
            if not 0 < tau < baseline_config.tau_days:
                raise ValueError("active_tau must be positive and smaller than the baseline tau.")
            model = DailyGateSpatialModel(
                grid, artifact=artifact,
                config=replace(baseline_config, active_tau_days=float(tau)))
            gains, active = [], []
            for t, observed in zip(origins, cells):
                p, diag = _predict(model, cat, t, it, ix)
                ll, count = spatial_log_likelihood(p, observed)
                gains.append(ll / count + np.log(n_cells) if count else np.nan)
                active.append(diag["active"])
            mean_gain = float(np.nanmean(gains))
            rows.append({
                "recent_days": float(recent_days), "active_tau_days": float(tau),
                "mean_gain": mean_gain,
                "delta_gain_vs_baseline": mean_gain - float(np.nanmean(base_gain)),
                "gate_active_fraction": float(np.mean(active)),
                "n_windows": len(origins),
                "n_scored_windows": int(np.isfinite(gains).sum()),
                "first_issue": origins[0], "last_label_end": origins[-1] + horizon,
            })
            if progress:
                progress(f"[gate tuning] recent={recent_days:g} d, tau={tau:g} d, "
                         f"gain={mean_gain:.6f}, active={np.mean(active):.1%}")
    ranking = pd.DataFrame(rows).sort_values(
        ["mean_gain", "active_tau_days", "recent_days"],
        ascending=[False, False, False], kind="stable").reset_index(drop=True)
    best = ranking.iloc[0]
    return ranking, artifacts[float(best["recent_days"])], float(best["active_tau_days"])


def evaluate_four_arms(catalog, grid, baseline_config, baseline_eval,
                       injection_times, injection_values, *, artifact,
                       active_tau_days, trigger_beta, distance_matrix, progress=print):
    import ot

    if not 0 < trigger_beta < 1:
        raise ValueError("The magnitude-trigger weight must be strictly between 0 and 1.")
    configs = {
        arm: replace(baseline_config, active_tau_days=active_tau_days,
                     beta=trigger_beta if arm in ("trigger", "gate_trigger") else 0.0)
        for arm in ARM_LABELS
    }
    models = {
        arm: DailyGateSpatialModel(
            grid, config=config,
            artifact=artifact if arm in ("gate", "gate_trigger") else None)
        for arm, config in configs.items()
    }
    n_cells = int(grid["n_grid"]) ** 2
    rows, predictions, maximum_error = [], {}, 0.0
    for arm, model in models.items():
        fields = []
        for ref in baseline_eval:
            issue, end = ref["t_issue"], ref["fw_end"]
            observed = _cells_in_window(catalog, grid, issue, end)
            if len(observed) != ref["K"]:
                raise ValueError("The observed catalog has changed; rerun the original spatial evaluation cells first.")
            p, diag = _predict(model, catalog, issue, injection_times, injection_values)
            if not np.all(np.isfinite(p)) or np.min(p) < 0 or not np.isclose(p.sum(), 1):
                raise ValueError("The spatial probability distribution is invalid.")
            if arm == "baseline":
                error = float(np.max(np.abs(p - ref["sw_fc"])))
                maximum_error = max(maximum_error, error)
                np.testing.assert_allclose(p, ref["sw_fc"], atol=1e-12, rtol=1e-10,
                                           err_msg="The baseline was not reproduced; rerun the original spatial evaluation cells.")
            fields.append(p)
            ll, count = spatial_log_likelihood(p, observed)
            row = {"arm": arm, "label": ARM_LABELS[arm], "k": ref["k"],
                   "t_issue": issue, "fw_end": end, "K": count,
                   "N_pred": ref["N_pred"], "LL": ll,
                   "gate_active": bool(diag["active"]),
                   "effective_tau_days": diag["tau_days_effective"]}
            for key in ("count_percentile", "shift_percentile", "injection_percentile",
                        "injection_at_issue", "injection_high"):
                row[key] = diag.get(key, np.nan)
            if count:
                obs = np.bincount(observed, minlength=n_cells).astype(float) / count
                order = np.argsort(p)[::-1]
                row.update(
                    gain=ll / count + np.log(n_cells),
                    KL=kl_divergence(p, obs), JS=js_divergence(p, obs),
                    EMD_km=float(ot.emd2(np.ascontiguousarray(p / p.sum()),
                                        np.ascontiguousarray(obs / obs.sum()),
                                        distance_matrix)),
                    hit10=float(obs[order[:max(1, round(.10 * n_cells))]].sum()),
                    hit20=float(obs[order[:max(1, round(.20 * n_cells))]].sum()),
                    spearman=float(spearmanr(p, obs)[0]),
                    probability_RMSE=float(np.sqrt(np.mean((p - obs) ** 2))),
                    count_RMSE=float(np.sqrt(np.mean((ref["N_pred"] * p - count * obs) ** 2))),
                )
            else:
                row.update({key: np.nan for key in (
                    "gain", "KL", "JS", "EMD_km", "hit10", "hit20", "spearman",
                    "probability_RMSE", "count_RMSE")})
            rows.append(row)
        predictions[arm] = np.stack(fields)
        if progress:
            progress(f"[spatial ablation] {ARM_LABELS[arm]} completed {len(fields)} windows")
    frame = pd.DataFrame(rows)
    base = frame.loc[frame.arm == "baseline", ["k", "gain"]].set_index("k")["gain"]
    frame["delta_gain"] = frame.gain - frame.k.map(base)
    summary = frame.groupby(["arm", "label"], sort=False).agg(
        n_windows=("k", "size"), n_scored_windows=("gain", "count"),
        mean_gain=("gain", "mean"), delta_gain=("delta_gain", "mean"),
        mean_JS=("JS", "mean"), mean_EMD_km=("EMD_km", "mean"),
        mean_hit10=("hit10", "mean"), mean_hit20=("hit20", "mean"),
        mean_probability_RMSE=("probability_RMSE", "mean"),
        mean_count_RMSE=("count_RMSE", "mean"),
        gate_active_fraction=("gate_active", "mean"),
    ).reset_index()
    valid = frame.loc[frame.gain.notna()].copy()
    valid["gain_win"] = valid.delta_gain > 1e-12
    valid["gain_tie"] = valid.delta_gain.abs() <= 1e-12
    rates = valid.groupby("arm").agg(
        gain_win_fraction=("gain_win", "mean"), gain_tie_fraction=("gain_tie", "mean"))
    summary = summary.join(rates, on="arm").sort_values(
        "mean_gain", ascending=False, kind="stable").reset_index(drop=True)
    summary.insert(0, "rank", summary.mean_gain.rank(method="min", ascending=False).astype(int))
    return frame, summary, predictions, {
        "baseline_max_abs_error": maximum_error,
        "configs": {arm: asdict(config) for arm, config in configs.items()},
    }

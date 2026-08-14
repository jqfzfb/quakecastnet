from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class BranchwiseTransportSpatialConfig:
    tau_days: float
    active_tau_days: float = 0.05
    variant: str = "branchwise_transport"
    kde_bandwidth_floor_km: float = 0.3
    kde_bandwidth_scale_km: float = 0.3
    n_eff_min: int = 2
    kde_cutoff_sigma: float = 6.0
    d0_km: float = 0.5
    alpha_n_ref: int = 200
    alpha_cap: float = 0.9
    beta: float = 0.3
    trigger_tau_days: float = 180.0
    drop_below_mc: bool = True
    trigger_r_max_km: float = 5.0
    gamma: float = 1.5
    eps: float = 1e-12
    mc_bin_width: float = 0.1
    max_pairwise_cells: int = 50_000_000


class BranchwiseTransportSpatialModel:

    RECENT_HOURS = 3.0
    N_SLICES = 4
    ROLLING_STRIDE_MINUTES = 2
    HORIZON_HOURS = 10.666666666666666
    VARIANTS = {"unsharpened", "mode_preserving", "branchwise_transport"}

    def __init__(self, sw_grid, artifact=None, stages=(), config=None, **kwargs):
        if config is None:
            config = BranchwiseTransportSpatialConfig(**kwargs)
        elif kwargs:
            config = config.__class__(**({**config.__dict__, **kwargs}))
        if config.variant not in self.VARIANTS:
            raise ValueError(f"unknown variant: {config.variant}")
        self.config = config
        self.artifact = _load_artifact(artifact)
        self.stages = tuple(sorted(
            (_coerce_stage(row) for row in stages), key=lambda row: row["order"]))

        self.n_grid = int(sw_grid["n_grid"])
        self.n_cells = self.n_grid ** 2
        self.lat_edges = np.asarray(sw_grid["lat_edges"], dtype=np.float64)
        self.lon_edges = np.asarray(sw_grid["lon_edges"], dtype=np.float64)
        lat_c = 0.5 * (self.lat_edges[:-1] + self.lat_edges[1:])
        lon_c = 0.5 * (self.lon_edges[:-1] + self.lon_edges[1:])
        lat, lon = np.meshgrid(lat_c, lon_c, indexing="ij")
        self.grid_lat = lat.ravel()
        self.grid_lon = lon.ravel()
        lat0 = float(np.median(lat_c))
        dx = float(np.median(np.diff(lon_c)) * np.cos(np.deg2rad(lat0)) * 111.32)
        dy = float(np.median(np.diff(lat_c)) * 111.32)
        self.grid_resolution_km = np.array([abs(dx), abs(dy)], dtype=np.float64)
        self._rolling_cache = None
        self._history_cache = None

    @classmethod
    def from_sw_grid(cls, grid, **kwargs):
        return cls(grid, **kwargs)

    def predict(self, lats, lons, mags, times, forecast_start, *,
                injection_times=None, injection_values=None,
                known_counts=None, return_diagnostics=False):
        la = np.asarray(lats, dtype=np.float64)
        lo = np.asarray(lons, dtype=np.float64)
        ma = np.asarray(mags, dtype=np.float64)
        tt = _dt(times)
        fs = np.datetime64(forecast_start, "ns")

        gate = self._gate(la, lo, tt, fs, injection_times, injection_values)
        active = bool(gate["active"])
        tau = self.config.active_tau_days if active else self.config.tau_days
        pre, old, state = self._five_step_fields(la, lo, ma, tt, fs, tau)
        diag = {**gate, "variant": self.config.variant,
                "tau_days_effective": float(tau), "n_modes": 0,
                "transport_used": False, "transport_weight": 0.0}
        if state is None or not active:
            out = old
            diag["reason"] = "inactive_exact_baseline"
            return (out, diag) if return_diagnostics else out

        mode = self._mode_state(state)
        if mode is None:
            out = pre
            diag["reason"] = "active_insufficient_recent_events"
            return (out, diag) if return_diagnostics else out
        diag["n_modes"] = int(mode["K"])

        if self.config.variant == "unsharpened":
            out = pre
            diag["reason"] = "active_unsharpened"
            return (out, diag) if return_diagnostics else out

        mode_field = self._mode_preserving_sharpen(pre, state, mode)
        if self.config.variant == "mode_preserving":
            diag["reason"] = "active_mode_preserving"
            return (mode_field, diag) if return_diagnostics else mode_field

        transport, transport_info = self._transport_field(
            state, mode, known_counts, fs, la, lo, ma, tt)
        diag.update(transport_info)
        if transport is None:
            diag["reason"] = "transport_unavailable_mode_fallback"
            return (mode_field, diag) if return_diagnostics else mode_field
        w_transport = self._expert_weight(
            fs, la, lo, ma, tt, injection_times, injection_values)
        out = self._norm(
            (1.0 - w_transport) * mode_field + w_transport * transport,
            self.n_cells)
        diag["transport_used"] = bool(w_transport > 0.0)
        diag["transport_weight"] = float(w_transport)
        diag["reason"] = "active_branchwise_transport"
        return (out, diag) if return_diagnostics else out

    def _five_step_fields(self, lats, lons, mags, times, fs, tau):
        c = self.config
        C = self.n_cells
        keep = times <= fs
        la, lo, ma, tt = (x[keep] for x in (lats, lons, mags, times))
        if not la.size:
            warnings.warn("BranchwiseTransportSpatial: no past events; uniform")
            u = np.full(C, 1.0 / C)
            return u, u, None
        age = np.clip(
            (fs - tt).astype("timedelta64[s]").astype(float) / 86400.0,
            0.0, None)
        lat0, lon0 = float(np.median(la)), float(np.median(lo))
        ev = _geo(la, lo, lat0, lon0)
        grid = _geo(self.grid_lat, self.grid_lon, lat0, lon0)
        weights = np.exp(-age / tau) if tau > 0 else np.zeros_like(age)
        ok = weights > c.eps
        neff = (weights[ok].sum() ** 2 / max(np.sum(weights[ok] ** 2), c.eps)
                if np.any(ok) else 0.0)
        if neff >= c.n_eff_min:
            h = max(c.kde_bandwidth_floor_km,
                    c.kde_bandwidth_scale_km * neff ** (-1.0 / 6.0))
            kde = self._norm(_gauss(
                ev[ok], weights[ok], grid, h, c.kde_cutoff_sigma,
                c.max_pairwise_cells), C)
        else:
            kde = np.full(C, 1.0 / C)
        phys = self._norm(
            1.0 / (np.linalg.norm(grid, axis=1) + c.d0_km) ** 1.5, C)
        alpha = min(c.alpha_cap, 0.3 + 0.6 * la.size / max(c.alpha_n_ref, 1))
        sw = self._norm(alpha * kde + (1.0 - alpha) * phys, C)
        b, mc = self._b_mc(ma)
        mok = ma >= mc if c.drop_below_mc else np.ones(ma.size, dtype=bool)
        if np.any(mok):
            ew = (np.exp(-age[mok] / c.trigger_tau_days)
                  * 10.0 ** (b * (ma[mok] - mc)))
            trig = self._norm(_trigger(
                ev[mok], ew, grid, c.d0_km ** 2, c.trigger_r_max_km,
                c.max_pairwise_cells), C)
        else:
            trig = np.full(C, 1.0 / C)
        pre = self._norm((1.0 - c.beta) * sw + c.beta * trig, C)
        old = self._norm(np.clip(pre, 0.0, None) ** c.gamma, C)
        state = {"fs": fs, "tt": tt, "la": la, "lo": lo, "ma": ma,
                 "ev": ev, "grid": grid, "lat0": lat0, "lon0": lon0}
        return pre, old, state

    def _mode_state(self, state):
        age_h = ((state["fs"] - state["tt"]).astype("timedelta64[s]")
                 .astype(float) / 3600.0)
        recent = age_h <= self.RECENT_HOURS
        points = state["ev"][recent]
        times = state["tt"][recent]
        if points.shape[0] < 2:
            return None
        fit = _select_gmm(points, self.grid_resolution_km)
        K = fit["K"]
        axis = self._stage_axis(state["fs"], state["lat0"], state["lon0"], points)
        order = _canonical_order(fit["means"], axis)
        fit = _reorder_gmm(fit, order)
        labels = np.argmax(_gmm_log_responsibility(points, fit), axis=1)
        return {"K": K, "fit": fit, "points": points, "times": times,
                "labels": labels, "axis": axis, "recent_mask": recent}

    def _mode_preserving_sharpen(self, pre, state, mode):
        if mode["K"] == 1:
            return pre.copy()
        grid_labels = np.argmax(
            _gmm_log_responsibility(state["grid"], mode["fit"]), axis=1)
        out = np.zeros_like(pre)
        for branch in range(mode["K"]):
            q = grid_labels == branch
            mass = float(pre[q].sum())
            shaped = np.clip(pre[q], 0.0, None) ** self.config.gamma
            total = shaped.sum()
            if total > self.config.eps:
                out[q] = mass * shaped / total
            else:
                out[q] = pre[q]
        return self._norm(out, self.n_cells)

    def _transport_field(self, state, mode, known_counts, fs,
                         lats, lons, mags, times):
        tracks = _track_modes(
            mode["points"], mode["times"], mode["K"], mode["axis"],
            self.grid_resolution_km, self.N_SLICES, fs)
        if tracks is None:
            return None, {"transport_reason": "tracking_failed"}
        history = self._history_records(fs, lats, lons, mags, times)
        prior = _history_prior(history, mode["K"])
        displacements, regime = self._forecast_displacements(
            tracks, prior, mode["K"], known_counts)
        if displacements is None:
            return None, {"transport_reason": "velocity_unavailable"}

        last = tracks["slices"][-1]
        valid_points = []
        valid_weights = []
        branch_counts = np.asarray(last["counts"], dtype=float)
        historical_ratio = _history_ratio(history, mode["K"])
        if branch_counts.sum() <= 0:
            return None, {"transport_reason": "empty_last_slice"}
        recent_ratio = branch_counts / branch_counts.sum()
        branch_ratio = 0.5 * (recent_ratio + historical_ratio)
        slice_ratio = _slice_weights(known_counts, self.N_SLICES)
        for lead in range(self.N_SLICES):
            for branch in range(mode["K"]):
                pts = last["points"][branch]
                if pts.size == 0:
                    continue
                moved = pts + displacements[lead, branch]
                valid_points.append(moved)
                valid_weights.append(np.full(
                    len(moved), slice_ratio[lead] * branch_ratio[branch]
                    / len(moved)))
        if not valid_points:
            return None, {"transport_reason": "no_transport_points"}
        points = np.concatenate(valid_points)
        weights = np.concatenate(valid_weights)
        field = self._norm(_gauss(
            points, weights, state["grid"],
            self.config.kde_bandwidth_floor_km,
            self.config.kde_cutoff_sigma,
            self.config.max_pairwise_cells), self.n_cells)
        return field, {
            "transport_reason": "ok", "transport_regime": regime,
            "branch_ratio": branch_ratio.tolist(),
            "transport_displacements_km": displacements.tolist(),
            "n_history_windows": len(history),
        }

    def _forecast_displacements(self, tracks, prior, K, known_counts):
        centers = tracks["centers"]
        observed = tracks["observed"]
        slice_times = tracks["time_hours"]
        last_time = float(slice_times[-1])
        leads = _forecast_lead_hours(
            known_counts, self.N_SLICES, self.HORIZON_HOURS)
        current = np.zeros((self.N_SLICES, K, 2), dtype=float)
        current_var = np.full((K, 2), np.inf)
        usable_branch = np.zeros(K, dtype=bool)
        for branch in range(K):
            q = observed[:, branch]
            if q.sum() < 2:
                continue
            t = slice_times[q]
            xy = centers[q, branch]
            velocity = np.array([_theil_sen(t, xy[:, j]) for j in range(2)])
            intercept = np.median(xy - t[:, None] * velocity, axis=0)
            residual = xy - (intercept + t[:, None] * velocity)
            current_var[branch] = np.maximum(
                np.var(residual, axis=0), (0.5 * self.grid_resolution_km) ** 2)
            current[:, branch] = ((leads - last_time)[:, None] * velocity)
            usable_branch[branch] = True
        if not np.any(usable_branch):
            return None, "none"

        gate = getattr(self, "_last_gate", None)
        regime = "shift" if gate and gate["shift_excess"] > gate["count_excess"] else "count"
        if prior is None:
            return current, regime
        out = current.copy()
        for lead in range(self.N_SLICES):
            for branch in range(K):
                if not usable_branch[branch]:
                    out[lead, branch] = prior["median"][lead, branch]
                    continue
                for axis in range(2):
                    vp = max(prior["variance"][lead, branch, axis],
                             (0.5 * self.grid_resolution_km[axis]) ** 2)
                    vc = max(current_var[branch, axis],
                             (0.5 * self.grid_resolution_km[axis]) ** 2)
                    out[lead, branch, axis] = (
                        current[lead, branch, axis] / vc
                        + prior["median"][lead, branch, axis] / vp
                    ) / (1.0 / vc + 1.0 / vp)
                    out[lead, branch, axis] = np.clip(
                        out[lead, branch, axis],
                        prior["q05"][lead, branch, axis],
                        prior["q95"][lead, branch, axis])
        return out, regime

    def _expert_weight(self, fs, lats, lons, mags, times,
                       injection_times, injection_values):
        records = self._history_records(fs, lats, lons, mags, times)
        scored = [row for row in records
                  if np.isfinite(row.get("mode_score", np.nan))
                  and np.isfinite(row.get("transport_score", np.nan))]
        groups = sorted({row["stage_group"] for row in scored})
        if len(groups) < 2:
            return 0.0
        scores = []
        for expert in ("mode_score", "transport_score"):
            per_group = [np.mean([row[expert] for row in scored
                                  if row["stage_group"] == group])
                         for group in groups]
            scores.append(float(np.mean(per_group)))
        shifted = np.asarray(scores) - max(scores)
        weight = np.exp(shifted)
        weight /= weight.sum()
        return float(weight[1])

    def _history_records(self, fs, lats, lons, mags, times):
        base = list(self.artifact.get("history", {}).get("records", []))
        cutoff = np.datetime64(self.artifact["seismic_cutoff"], "ns")
        if fs <= cutoff:
            return [row for row in base
                    if np.datetime64(row["target_end"], "ns") <= fs]
        key = (_catalog_hash(lats, lons, times), int(fs.astype("int64")))
        if self._history_cache is not None and self._history_cache[0] == key:
            return self._history_cache[1]
        records = list(base)
        if records:
            origin = np.datetime64(records[-1]["origin"], "ns")
        else:
            origin = np.datetime64(self.artifact["history"]["first_origin"], "ns")
            origin -= _hours(self.HORIZON_HOURS)
        horizon = _hours(self.HORIZON_HOURS)
        origin += horizon
        while origin + horizon <= fs:
            row = self._make_history_record(
                origin, lats, lons, mags, times, records)
            if row is not None:
                records.append(row)
            origin += horizon
        self._history_cache = (key, records)
        return records

    def _make_history_record(self, origin, lats, lons, mags, times, previous):
        past = times <= origin
        target = (times > origin) & (times <= origin + _hours(self.HORIZON_HOURS))
        if past.sum() < 2 or target.sum() < 2:
            return None
        pre, _, state = self._five_step_fields(
            lats[past], lons[past], mags[past], times[past], origin,
            self.config.active_tau_days)
        mode = self._mode_state(state)
        if mode is None:
            return None
        mode_field = self._mode_preserving_sharpen(pre, state, mode)
        target_xy = _geo(
            lats[target], lons[target], state["lat0"], state["lon0"])
        record = _branch_transition_record(
            origin, origin + _hours(self.HORIZON_HOURS), mode,
            target_xy, times[target], self.grid_resolution_km,
            self.N_SLICES, self._stage_group(origin))
        if record is None:
            return None
        cells = _event_cells(
            lats[target], lons[target], self.lat_edges, self.lon_edges,
            self.n_grid)
        if cells.size:
            record["mode_score"] = float(np.log(
                np.clip(mode_field[cells], self.config.eps, None)).mean())
        if previous:
            prior = _history_prior(previous, mode["K"])
            transport = self._historical_transport_field(state, mode, prior)
            if transport is not None and cells.size:
                record["transport_score"] = float(np.log(
                    np.clip(transport[cells], self.config.eps, None)).mean())
        return record

    def _historical_transport_field(self, state, mode, prior):
        tracks = _track_modes(
            mode["points"], mode["times"], mode["K"], mode["axis"],
            self.grid_resolution_km, self.N_SLICES, state["fs"])
        if tracks is None:
            return None
        displacements, _ = self._forecast_displacements(
            tracks, prior, mode["K"], None)
        if displacements is None:
            return None
        last = tracks["slices"][-1]
        counts = np.asarray(last["counts"], float)
        if counts.sum() <= 0:
            return None
        recent_ratio = counts / counts.sum()
        historical_ratio = (_history_ratio([], mode["K"])
                            if prior is None else recent_ratio)
        ratio = 0.5 * (recent_ratio + historical_ratio)
        ratio = ratio / ratio.sum()
        points, weights = [], []
        for lead in range(self.N_SLICES):
            for branch in range(mode["K"]):
                pts = last["points"][branch]
                if not len(pts):
                    continue
                points.append(pts + displacements[lead, branch])
                weights.append(np.full(
                    len(pts), ratio[branch] / (self.N_SLICES * len(pts))))
        if not points:
            return None
        return self._norm(_gauss(
            np.concatenate(points), np.concatenate(weights), state["grid"],
            self.config.kde_bandwidth_floor_km,
            self.config.kde_cutoff_sigma,
            self.config.max_pairwise_cells), self.n_cells)

    def _stage_axis(self, fs, lat0, lon0, points):
        available = [row for row in self.stages if row["knowledge_time"] <= fs]
        if len(available) >= 2:
            starts = [row for row in available if row["start"] <= fs]
            idx = max(1, len(starts) - 1)
            pair = available[max(0, idx - 1):idx + 1]
            if len(pair) == 2:
                xy = _geo(
                    np.array([row["lat"] for row in pair]),
                    np.array([row["lon"] for row in pair]), lat0, lon0)
                axis = xy[1] - xy[0]
                if np.linalg.norm(axis) > self.config.eps:
                    return axis / np.linalg.norm(axis)
        centered = points - np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        if axis[0] < 0:
            axis = -axis
        return axis

    def _stage_group(self, origin):
        available = [row for row in self.stages if row["knowledge_time"] <= origin]
        started = [row for row in available if row["start"] <= origin]
        return started[-1]["stage_id"] if started else "pre-stage"

    def _gate(self, lats, lons, times, fs, injection_times, injection_values):
        inactive = {"active": False, "count_excess": 0.0,
                    "shift_excess": 0.0, "injection_high": False}
        if not self.artifact or injection_times is None or injection_values is None:
            self._last_gate = inactive
            return inactive
        endpoints, counts, shifts = self._rolling_activity(lats, lons, times, fs)
        q = endpoints <= fs
        if not np.any(q):
            self._last_gate = inactive
            return inactive
        counts, shifts = counts[q], shifts[q]
        valid_shift = shifts[np.isfinite(shifts)]
        p_count = _ecdf(counts, counts[-1])
        p_shift = _ecdf(valid_shift, shifts[-1]) if valid_shift.size else 0.0
        tc = float(self.artifact["gate"]["count_threshold"])
        ts = float(self.artifact["gate"]["shift_threshold"])
        ec, es = _excess(p_count, tc), _excess(p_shift, ts)

        it = _dt(injection_times)
        ix = np.asarray(injection_values, dtype=float).reshape(-1)
        finite = np.isfinite(ix)
        it, ix = it[finite], ix[finite]
        order = np.argsort(it)
        it, ix = it[order], ix[order]
        j = int(np.searchsorted(it, fs, side="right") - 1)
        issue_inj = float(ix[j]) if j >= 0 else 0.0
        ref = np.asarray(self.artifact["injection"]["positive_reference"], float)
        p_inj = _ecdf(ref, issue_inj) if issue_inj > 0 else 0.0
        injection_high = bool(
            issue_inj > 0 and p_inj > self.artifact["injection"]["threshold"])
        result = {
            "active": bool(max(ec, es) > 0 and injection_high),
            "count_percentile": float(p_count),
            "shift_percentile": float(p_shift),
            "count_excess": float(ec), "shift_excess": float(es),
            "injection_at_issue": issue_inj,
            "injection_percentile": float(p_inj),
            "injection_high": injection_high,
        }
        self._last_gate = result
        return result

    def _rolling_activity(self, lats, lons, times, fs):
        anchor = np.datetime64(self.artifact["rolling"]["first_endpoint"], "ns")
        stride = np.timedelta64(
            int(self.artifact["rolling"]["stride_minutes"]), "m")
        key = _catalog_hash(lats, lons, times)
        cache = self._rolling_cache
        if cache is None or cache["key"] != key:
            end = np.empty(0, dtype="datetime64[ns]")
            counts = np.empty(0, dtype=int)
            shifts = np.empty(0, dtype=float)
            next_end = anchor
        else:
            end, counts, shifts = cache["end"], cache["counts"], cache["shifts"]
            next_end = end[-1] + stride if end.size else anchor
        if fs >= next_end:
            n = int((fs - next_end) / stride) + 1
            new_end = next_end + np.arange(n) * stride
            nc, ns = _rolling_stats(
                lats, lons, times, new_end, self.RECENT_HOURS,
                self.artifact["projection"]["lat0"],
                self.artifact["projection"]["lon0"])
            end = np.r_[end, new_end]
            counts = np.r_[counts, nc]
            shifts = np.r_[shifts, ns]
            self._rolling_cache = {"key": key, "end": end,
                                   "counts": counts, "shifts": shifts}
        return end, counts, shifts

    def _b_mc(self, mags):
        c = self.config
        m = np.asarray(mags, dtype=float)
        m = m[np.isfinite(m)]
        if m.size < 2:
            return 1.0, float(np.min(m)) if m.size else 0.0
        bins = np.arange(np.floor(m.min()), np.ceil(m.max()) + 2 * c.mc_bin_width,
                         c.mc_bin_width)
        hist, edges = np.histogram(m, bins)
        mc = float(edges[int(np.argmax(hist))])
        above = m[m >= mc]
        if above.size < 2 or above.mean() <= mc:
            return 1.0, mc
        return float(np.log10(np.e) / (above.mean() - mc)), mc

    @staticmethod
    def _norm(values, cells, eps=1e-12):
        values = np.clip(np.asarray(values, dtype=float), 0.0, None)
        total = values.sum()
        if not np.isfinite(total) or total <= eps:
            return np.full(cells, 1.0 / cells)
        return values / total


def build_branchwise_transport_artifact(sw_grid, lats, lons, mags, times,
                                        cutoff, injection_times,
                                        injection_values, stages=()):
    la = np.asarray(lats, float); lo = np.asarray(lons, float)
    ma = np.asarray(mags, float); tt = _dt(times)
    cutoff = np.datetime64(cutoff, "ns")
    keep = tt <= cutoff
    if keep.sum() < 2:
        raise ValueError("insufficient pre-cutoff earthquakes")
    it = _dt(injection_times); ix = np.asarray(injection_values, float).reshape(-1)
    finite = np.isfinite(ix); it, ix = it[finite], ix[finite]
    first_endpoint = np.min(it) + np.timedelta64(6, "h")
    stride = np.timedelta64(BranchwiseTransportSpatialModel.ROLLING_STRIDE_MINUTES, "m")
    n = int((cutoff - first_endpoint) / stride) + 1
    endpoints = first_endpoint + np.arange(n) * stride
    lat0, lon0 = float(np.median(la[keep])), float(np.median(lo[keep]))
    counts, shifts = _rolling_stats(
        la[keep], lo[keep], tt[keep], endpoints,
        BranchwiseTransportSpatialModel.RECENT_HOURS, lat0, lon0)
    positive = np.sort(ix[ix > 0])
    artifact = {
        "schema": "utahforge-branchwise-transport-v1",
        "seismic_cutoff": str(np.datetime_as_string(cutoff, unit="ns")),
        "projection": {"lat0": lat0, "lon0": lon0},
        "rolling": {
            "window_hours": BranchwiseTransportSpatialModel.RECENT_HOURS,
            "stride_minutes": BranchwiseTransportSpatialModel.ROLLING_STRIDE_MINUTES,
            "first_endpoint": str(np.datetime_as_string(first_endpoint, unit="ns")),
            "n_reference_endpoints": int(n),
        },
        "gate": {
            "count_threshold": _median_mad_percentile(counts),
            "shift_threshold": _median_mad_percentile(shifts[np.isfinite(shifts)]),
        },
        "injection": {
            "threshold": _median_mad_percentile(positive),
            "positive_reference": positive.tolist(),
            "scenario_hash": _injection_hash(it, ix),
        },
        "catalog_hash_at_cutoff": _catalog_hash(la[keep], lo[keep], tt[keep]),
        "history": {
            "source_hours": BranchwiseTransportSpatialModel.RECENT_HOURS,
            "horizon_hours": BranchwiseTransportSpatialModel.HORIZON_HOURS,
            "spacing_hours": BranchwiseTransportSpatialModel.HORIZON_HOURS,
            "n_slices": BranchwiseTransportSpatialModel.N_SLICES,
            "records": [],
        },
    }
    model = BranchwiseTransportSpatialModel(
        sw_grid, artifact=artifact, stages=stages,
        config=BranchwiseTransportSpatialConfig(
            tau_days=0.5, active_tau_days=0.05,
            kde_bandwidth_floor_km=0.05,
            kde_bandwidth_scale_km=0.05, gamma=1.5, beta=0.3))
    horizon = _hours(model.HORIZON_HOURS)
    first_origin = np.min(tt[keep]) + horizon
    artifact["history"]["first_origin"] = str(
        np.datetime_as_string(first_origin, unit="ns"))
    records = []
    origin = first_origin
    while origin + horizon <= cutoff:
        row = model._make_history_record(
            origin, la[keep], lo[keep], ma[keep], tt[keep], records)
        if row is not None:
            records.append(row)
        origin += horizon
    artifact["history"]["records"] = records
    artifact["history"]["n_effective_windows"] = len(records)
    artifact["history"]["n_stage_groups"] = len(
        {row["stage_group"] for row in records})
    return artifact


def _select_gmm(points, resolution):
    one = _fit_tied_gmm(points, 1, resolution)
    if len(points) < 2:
        return one
    two = _fit_tied_gmm(points, 2, resolution)
    return two if two["bic"] < one["bic"] else one


def _fit_tied_gmm(points, K, resolution, initial_means=None):
    x = np.asarray(points, float)
    n, dim = x.shape
    floor = np.diag((0.5 * np.asarray(resolution, float)) ** 2)
    if K == 1:
        means = np.mean(x, axis=0, keepdims=True)
        weights = np.ones(1)
    else:
        means = (_farthest_pair(x) if initial_means is None
                 else np.asarray(initial_means, float).copy())
        weights = np.full(K, 1.0 / K)
    cov = np.cov(x.T, bias=True) if n > 1 else floor.copy()
    cov = np.atleast_2d(cov) + floor
    previous = -np.inf
    for _ in range(100):
        fit = {"means": means, "weights": weights, "cov": cov, "K": K}
        log_resp = _gmm_log_responsibility(x, fit)
        row_max = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - row_max)
        denom = resp.sum(axis=1, keepdims=True)
        resp /= np.maximum(denom, 1e-300)
        nk = np.maximum(resp.sum(axis=0), 1e-9)
        weights = nk / n
        means = (resp.T @ x) / nk[:, None]
        cov = floor.copy()
        for k in range(K):
            diff = x - means[k]
            cov += (resp[:, k, None] * diff).T @ diff / n
        loglike = float(np.sum(row_max[:, 0] + np.log(np.maximum(denom[:, 0], 1e-300))))
        if abs(loglike - previous) <= 1e-9 * (1.0 + abs(loglike)):
            break
        previous = loglike
    params = K * dim + dim * (dim + 1) / 2 + K - 1
    return {"means": means, "weights": weights, "cov": cov, "K": K,
            "loglike": loglike, "bic": float(-2 * loglike + params * np.log(n))}


def _gmm_log_responsibility(points, fit):
    x = np.asarray(points, float)
    inv = np.linalg.inv(fit["cov"])
    sign, logdet = np.linalg.slogdet(fit["cov"])
    if sign <= 0:
        raise np.linalg.LinAlgError("non-positive GMM covariance")
    out = []
    for mean, weight in zip(fit["means"], fit["weights"]):
        diff = x - mean
        q = np.einsum("ni,ij,nj->n", diff, inv, diff)
        out.append(np.log(max(weight, 1e-300)) - 0.5 * (
            x.shape[1] * np.log(2 * np.pi) + logdet + q))
    return np.stack(out, axis=1)


def _farthest_pair(points):
    diff = points[:, None, :] - points[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return np.stack([points[i], points[j]])


def _canonical_order(means, axis):
    cross = np.array([-axis[1], axis[0]])
    return np.lexsort((means @ axis, means @ cross))


def _reorder_gmm(fit, order):
    return {**fit, "means": fit["means"][order], "weights": fit["weights"][order]}


def _track_modes(points, times, K, axis, resolution, n_slices, forecast_start):
    order = np.argsort(times)
    points, times = points[order], times[order]
    chunks = np.array_split(np.arange(len(points)), n_slices)
    slices = []
    previous = None
    for indices in chunks:
        if len(indices) == 0:
            return None
        pts = points[indices]
        observed = np.zeros(K, dtype=bool)
        branch_points = [np.empty((0, 2)) for _ in range(K)]
        if len(pts) >= K:
            fit = _fit_tied_gmm(pts, K, resolution, previous)
            labels = np.argmax(_gmm_log_responsibility(pts, fit), axis=1)
            centers = fit["means"].copy()
            if previous is None:
                assignment = _canonical_order(centers, axis)
            else:
                rows, cols = linear_sum_assignment(
                    np.linalg.norm(previous[:, None] - centers[None, :], axis=2))
                assignment = np.empty(K, dtype=int); assignment[rows] = cols
            centers = centers[assignment]
            for branch, component in enumerate(assignment):
                q = labels == component
                if np.any(q):
                    branch_points[branch] = pts[q]
                    centers[branch] = np.median(pts[q], axis=0)
                    observed[branch] = True
                elif previous is not None:
                    centers[branch] = previous[branch]
        elif previous is not None:
            centers = previous.copy()
            nearest = int(np.argmin(np.linalg.norm(previous - pts[0], axis=1)))
            centers[nearest] = pts[0]
            branch_points[nearest] = pts.copy()
            observed[nearest] = True
        else:
            return None
        previous = centers.copy()
        slices.append({"centers": centers, "observed": observed,
                       "points": branch_points,
                       "counts": [len(x) for x in branch_points],
                       "time": times[indices]})
    time_hours = np.array([
        float(np.median((row["time"] - forecast_start)
                        .astype("timedelta64[s]").astype(float)) / 3600)
        for row in slices])
    return {"slices": slices,
            "centers": np.stack([row["centers"] for row in slices]),
            "observed": np.stack([row["observed"] for row in slices]),
            "time_hours": time_hours}


def _branch_transition_record(origin, target_end, source_mode, target_points,
                              target_times, resolution, n_slices, stage_group):
    K = source_mode["K"]
    chunks = np.array_split(np.argsort(target_times), n_slices)
    displacements = np.zeros((n_slices, K, 2), dtype=float)
    ratios = np.zeros((n_slices, K), dtype=float)
    previous = source_mode["fit"]["means"].copy()
    for lead, indices in enumerate(chunks):
        if len(indices) < K:
            return None
        fit = _fit_tied_gmm(target_points[indices], K, resolution, previous)
        labels = np.argmax(_gmm_log_responsibility(target_points[indices], fit), axis=1)
        rows, cols = linear_sum_assignment(
            np.linalg.norm(previous[:, None] - fit["means"][None, :], axis=2))
        assignment = np.empty(K, dtype=int); assignment[rows] = cols
        centers = fit["means"][assignment]
        for branch, component in enumerate(assignment):
            q = labels == component
            if np.any(q):
                centers[branch] = np.median(target_points[indices][q], axis=0)
                ratios[lead, branch] = q.mean()
        displacements[lead] = centers - source_mode["fit"]["means"]
        previous = centers
    return {
        "origin": str(np.datetime_as_string(origin, unit="ns")),
        "target_end": str(np.datetime_as_string(target_end, unit="ns")),
        "stage_group": stage_group, "K": int(K),
        "displacements_km": displacements.tolist(),
        "branch_ratios": ratios.tolist(),
    }


def _history_prior(records, K):
    rows = [row for row in records if int(row["K"]) == K]
    if not rows:
        return None
    values = np.asarray([row["displacements_km"] for row in rows], float)
    return {"median": np.median(values, axis=0),
            "variance": np.var(values, axis=0),
            "q05": np.quantile(values, 0.05, axis=0),
            "q95": np.quantile(values, 0.95, axis=0)}


def _history_ratio(records, K):
    rows = [row for row in records if int(row["K"]) == K]
    if not rows:
        return np.full(K, 1.0 / K)
    ratios = np.asarray([row["branch_ratios"][-1] for row in rows], float)
    result = np.mean(ratios, axis=0)
    return result / result.sum() if result.sum() > 0 else np.full(K, 1.0 / K)


def _theil_sen(t, y):
    slopes = []
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if t[j] != t[i]:
                slopes.append((y[j] - y[i]) / (t[j] - t[i]))
    return float(np.median(slopes)) if slopes else 0.0


def _slice_weights(known_counts, n_slices):
    if known_counts is None:
        return np.full(n_slices, 1.0 / n_slices)
    values = np.clip(np.asarray(known_counts, float).reshape(-1), 0.0, None)
    chunks = np.array_split(values, n_slices)
    weights = np.asarray([chunk.sum() for chunk in chunks], float)
    return weights / weights.sum() if weights.sum() > 0 else np.full(n_slices, 1.0 / n_slices)


def _forecast_lead_hours(known_counts, n_slices, horizon_hours):
    if known_counts is None:
        return (np.arange(n_slices) + 0.5) * horizon_hours / n_slices
    values = np.clip(np.asarray(known_counts, float).reshape(-1), 0.0, None)
    chunks = np.array_split(np.arange(len(values)), n_slices)
    dt = horizon_hours / max(len(values), 1)
    result = []
    for indices in chunks:
        weights = values[indices]
        centers = (indices + 0.5) * dt
        result.append(float(np.average(centers, weights=weights))
                      if weights.sum() > 0 else float(np.mean(centers)))
    return np.asarray(result)


def _coerce_stage(row):
    return {"stage_id": str(row["stage_id"]), "order": int(row["order"]),
            "lat": float(row["lat"]), "lon": float(row["lon"]),
            "start": np.datetime64(row["start"], "ns"),
            "end": np.datetime64(row["end"], "ns"),
            "knowledge_time": np.datetime64(row["knowledge_time"], "ns")}


def _load_artifact(value):
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        with Path(value).open(encoding="utf-8") as handle:
            value = json.load(handle)
    if value.get("schema") != "utahforge-branchwise-transport-v1":
        raise ValueError("unsupported branchwise transport artifact")
    return value


def _rolling_stats(lats, lons, times, endpoints, hours, lat0, lon0):
    order = np.argsort(times); tt = times[order]
    xy = _geo(np.asarray(lats)[order], np.asarray(lons)[order], lat0, lon0)
    span = _hours(hours)
    counts = np.empty(len(endpoints), int)
    shifts = np.full(len(endpoints), np.nan)
    for i, end in enumerate(endpoints):
        middle, start = end - span, end - 2 * span
        a = np.searchsorted(tt, start, side="right")
        b = np.searchsorted(tt, middle, side="right")
        c = np.searchsorted(tt, end, side="right")
        counts[i] = c - b
        if b > a and c > b:
            shifts[i] = np.linalg.norm(
                np.median(xy[b:c], axis=0) - np.median(xy[a:b], axis=0))
    return counts, shifts


def _median_mad_percentile(values):
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values):
        return 1.0
    p = np.searchsorted(np.sort(values), values, side="right") / len(values)
    median = np.median(p)
    return float(min(1.0, median + np.median(np.abs(p - median))))


def _ecdf(reference, value):
    ref = np.asarray(reference, float); ref = ref[np.isfinite(ref)]
    return float(np.searchsorted(np.sort(ref), value, side="right") / len(ref)) if len(ref) else 0.0


def _excess(p, threshold):
    return float((p - threshold) / (1.0 - threshold)) if p > threshold and threshold < 1 else 0.0


def _event_cells(lats, lons, lat_edges, lon_edges, n_grid):
    iy = np.searchsorted(lat_edges, lats, side="right") - 1
    ix = np.searchsorted(lon_edges, lons, side="right") - 1
    q = (iy >= 0) & (iy < n_grid) & (ix >= 0) & (ix < n_grid)
    return (iy[q] * n_grid + ix[q]).astype(int)


def _hours(value):
    return np.timedelta64(int(round(float(value) * 3600)), "s")


def _dt(values):
    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]")
    out = np.empty(len(arr), dtype="datetime64[ns]")
    for i, value in enumerate(arr):
        out[i] = np.datetime64(value, "ns")
    return out


def _geo(lats, lons, lat0, lon0):
    return np.stack([
        (np.asarray(lons, float) - lon0) * np.cos(np.deg2rad(lat0)) * 111.32,
        (np.asarray(lats, float) - lat0) * 111.32], axis=-1)


def _gauss(events, weights, grid, bandwidth, cutoff_sigma, max_pairs):
    out = np.zeros(len(grid)); cutoff2 = (cutoff_sigma * bandwidth) ** 2
    block = max(1, int(max_pairs // max(len(grid), 1)))
    for start in range(0, len(events), block):
        diff = events[start:start + block, None] - grid[None]
        d2 = np.sum(diff * diff, axis=-1)
        d2 = np.where(d2 > cutoff2, np.inf, d2)
        out += (weights[start:start + block, None]
                * np.exp(-d2 / (2 * bandwidth ** 2))).sum(axis=0)
    return out


def _trigger(events, weights, grid, d0_sq, rmax, max_pairs):
    out = np.zeros(len(grid)); rmax2 = rmax ** 2
    block = max(1, int(max_pairs // max(len(grid), 1)))
    for start in range(0, len(events), block):
        diff = events[start:start + block, None] - grid[None]
        d2 = np.sum(diff * diff, axis=-1)
        contrib = weights[start:start + block, None] / (d2 + d0_sq) ** 1.5
        out += np.where(d2 > rmax2, 0.0, contrib).sum(axis=0)
    return out


def _catalog_hash(lats, lons, times):
    digest = sha256()
    for arr in (np.asarray(lats, dtype="<f8"), np.asarray(lons, dtype="<f8"),
                _dt(times).astype("<i8")):
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def _injection_hash(times, values):
    digest = sha256()
    digest.update(np.ascontiguousarray(_dt(times).astype("<i8")).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())
    return digest.hexdigest()

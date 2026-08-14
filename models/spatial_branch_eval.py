from __future__ import annotations

from dataclasses import asdict
import warnings

import numpy as np
import pandas as pd


def load_catalog(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    return {
        "lats": df["latitude"].to_numpy(dtype=np.float64),
        "lons": df["longitude"].to_numpy(dtype=np.float64),
        "mags": df["magnitude"].to_numpy(dtype=np.float64),
        "times": pd.to_datetime(df["time"].to_numpy()).astype("datetime64[ns]"),
    }


def load_dataset(npy_path: str) -> dict:
    return np.load(npy_path, allow_pickle=True).item()


def build_sw_grid(lats, lons, n_grid: int) -> dict:
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    lat_lo, lat_hi = np.quantile(lats, [0.005, 0.995])
    lon_lo, lon_hi = np.quantile(lons, [0.005, 0.995])
    lp = 0.2 * (lat_hi - lat_lo)
    lop = 0.2 * (lon_hi - lon_lo)
    return {
        "n_grid": int(n_grid),
        "lat_edges": np.linspace(lat_lo - lp, lat_hi + lp, n_grid + 1),
        "lon_edges": np.linspace(lon_lo - lop, lon_hi + lop, n_grid + 1),
    }


def events_to_cells(lats, lons, lat_edges, lon_edges, n_grid: int) -> np.ndarray:
    lat = np.asarray(lats, dtype=np.float64)
    lon = np.asarray(lons, dtype=np.float64)
    la = np.searchsorted(lat_edges, lat, side="right") - 1
    lo = np.searchsorted(lon_edges, lon, side="right") - 1
    cell = la * n_grid + lo
    oob = (la < 0) | (la >= n_grid) | (lo < 0) | (lo >= n_grid)
    cell[oob] = -1
    return cell


def build_observed_sw(catalog: dict, sw_grid: dict, time_axis, window_points: int) -> np.ndarray:
    n_grid = int(sw_grid["n_grid"])
    n_cells = n_grid * n_grid
    lat_e = np.asarray(sw_grid["lat_edges"], dtype=np.float64)
    lon_e = np.asarray(sw_grid["lon_edges"], dtype=np.float64)

    ev_lat = catalog["lats"]; ev_lon = catalog["lons"]
    ev_t = catalog["times"]
    order = np.argsort(ev_t)
    ev_t_s = ev_t[order]
    ev_cell = events_to_cells(ev_lat, ev_lon, lat_e, lon_e, n_grid)
    ev_cell_s = ev_cell[order]
    valid = ev_cell_s >= 0
    ev_t_s = ev_t_s[valid]
    ev_cell_s = ev_cell_s[valid]

    time_axis = np.asarray(time_axis).astype("datetime64[ns]")
    T = len(time_axis)
    dt = (time_axis[1] - time_axis[0]) if T >= 2 else np.timedelta64(1, "D")
    W = int(window_points)
    uniform = np.full(n_cells, 1.0 / n_cells)
    out = np.zeros((T, n_cells), dtype=np.float64)

    ev_t_s_int = ev_t_s.astype("datetime64[ns]")
    for i in range(T):
        t = time_axis[i]
        t0 = t - (W - 1) * dt
        t1 = t + dt
        lo = np.searchsorted(ev_t_s_int, np.datetime64(t0, "ns"), side="left")
        hi = np.searchsorted(ev_t_s_int, np.datetime64(t1, "ns"), side="left")
        cells = ev_cell_s[lo:hi]
        if cells.size:
            counts = np.bincount(cells, minlength=n_cells).astype(np.float64)
            out[i] = counts / counts.sum()
        else:
            out[i] = uniform
    return out


def uniform_reference(n_cells: int) -> np.ndarray:
    return np.full(n_cells, 1.0 / n_cells)


def spatial_log_likelihood(sw_forecast, event_cells, eps: float = 1e-12) -> tuple:
    cells = np.asarray(event_cells)
    cells = cells[cells >= 0]
    if cells.size == 0:
        return 0.0, 0
    return float(np.sum(np.log(np.asarray(sw_forecast)[cells] + eps))), int(cells.size)


def kl_divergence(p, q, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, None)
    p = p / max(p.sum(), eps); q = q / max(q.sum(), eps)
    mask = p > 0
    return float(np.sum(p[mask] * np.log((p[mask] + eps) / (q[mask] + eps))))


def js_divergence(p, q, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    q = np.clip(np.asarray(q, dtype=np.float64), 0.0, None)
    p = p / max(p.sum(), eps); q = q / max(q.sum(), eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)

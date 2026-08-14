from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import curve_fit

from .tft_model_new import TemporalFusionTransformerNew as TemporalFusionTransformer

STEP_SECONDS = 480
_EPS = 1e-12


@dataclass
class ShutInEpisode:

    start: int
    end: int
    rates: np.ndarray
    start_level: float

    def normalized_points(self, step_hours: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.start_level <= 0 or len(self.rates) == 0:
            return np.empty(0), np.empty(0)
        dt = (np.arange(1, len(self.rates) + 1)) * step_hours
        return dt, self.rates / self.start_level


def _robust_start_level(rates: np.ndarray) -> float:
    k = min(5, len(rates))
    lvl = float(np.mean(rates[:k])) if k else 0.0
    if lvl <= 0:
        nz = rates[rates > 0]
        lvl = float(np.median(nz)) if len(nz) else 0.0
    return lvl


def extract_shutin_episodes(
    inj_raw: Sequence[float],
    seis_raw: Sequence[float],
    threshold_bbl: float,
    step_hours: float,
) -> List[ShutInEpisode]:
    inj = np.asarray(inj_raw, dtype=float)
    seis = np.asarray(seis_raw, dtype=float)
    is_shutin = inj <= threshold_bbl
    episodes: List[ShutInEpisode] = []
    i, n = 0, len(inj)
    while i < n:
        if not is_shutin[i]:
            i += 1
            continue
        j = i
        while j < n and is_shutin[j]:
            j += 1
        episodes.append(ShutInEpisode(i, j, seis[i:j].copy(), _robust_start_level(seis[i:j])))
        i = j
    return episodes


def double_omori(dt, w, c1, p1, c2, p2):
    dt = np.asarray(dt, dtype=float)
    return w * (c1 / (dt + c1)) ** p1 + (1.0 - w) * (c2 / (dt + c2)) ** p2


def single_omori(dt, c, p):
    dt = np.asarray(dt, dtype=float)
    return (c / (dt + c)) ** p


@dataclass
class DecayShape:

    kind: str
    params: Tuple[float, ...] = ()
    scale: float = 1.0

    @property
    def params_vector(self) -> np.ndarray:
        return np.asarray(self.params, dtype=float)

    def __call__(self, dt) -> np.ndarray:
        dt = np.atleast_1d(np.asarray(dt, dtype=float))
        if self.kind == "double":
            return self.scale * double_omori(dt, *self.params)
        if self.kind == "single":
            return self.scale * single_omori(dt, *self.params)
        return np.ones_like(dt)


def _try_fit(func, dt, v, p0, bounds, sigma=None) -> Tuple[Optional[tuple], float]:
    try:
        popt, _ = curve_fit(func, dt, v, p0=p0, bounds=bounds, maxfev=20000, sigma=sigma)
        resid = func(dt, *popt) - v
        wgt = np.ones_like(resid) if sigma is None else 1.0 / np.asarray(sigma) ** 2
        rmse = float(np.sqrt(np.average(resid ** 2, weights=wgt)))
        return tuple(float(x) for x in popt), rmse
    except Exception:
        return None, np.inf


def fit_decay_shape(
    dt_hours: Sequence[float],
    values: Sequence[float],
    min_points: int = 8,
    sigma: Optional[Sequence[float]] = None,
) -> DecayShape:
    dt = np.asarray(dt_hours, dtype=float)
    v = np.asarray(values, dtype=float)
    m = np.isfinite(dt) & np.isfinite(v) & (dt > 0)
    dt, v = dt[m], np.clip(v[m], 0.0, 3.0)
    sig = None if sigma is None else np.asarray(sigma, dtype=float)[m]
    if len(dt) < min_points:
        return DecayShape("identity")

    def _double(dt, s, w, c1, p1, c2, p2):
        return s * double_omori(dt, w, c1, p1, c2, p2)

    def _single(dt, s, c, p):
        return s * single_omori(dt, c, p)

    popt_d, rmse_d = _try_fit(
        _double, dt, v,
        p0=[1.0, 0.6, 0.3, 1.0, 8.0, 1.5],
        bounds=([1e-3, 0.0, 1e-3, 0.05, 1e-3, 0.05], [10.0, 1.0, 200.0, 5.0, 200.0, 5.0]),
        sigma=sig,
    )
    popt_s, rmse_s = _try_fit(
        _single, dt, v,
        p0=[1.0, 1.0, 1.0],
        bounds=([1e-3, 1e-3, 0.05], [10.0, 200.0, 5.0]),
        sigma=sig,
    )

    if popt_d is not None and rmse_d <= rmse_s * 1.05:
        s, w, c1, p1, c2, p2 = popt_d
        return DecayShape("double", (w, c1, p1, c2, p2), scale=s)
    if popt_s is not None:
        s, c, p = popt_s
        return DecayShape("single", (c, p), scale=s)
    return DecayShape("identity")


@dataclass
class WindowCorrection:

    g: DecayShape
    a_ep: Optional[float]
    dt0_hours: float
    w: float = 1.0
    quiet: bool = False
    floor_raw: float = 0.0
    dt_val_hours: float = np.inf
    misfit: Optional[float] = None
    pool_scale: Optional[float] = None


class ShutInDecayEstimator:

    def __init__(
        self,
        threshold_bbl: float = 1.0,
        step_hours: float = STEP_SECONDS / 3600.0,
        min_points: int = 8,
        detection_quantum: Optional[float] = None,
    ):
        self.threshold_bbl = float(threshold_bbl)
        self.step_hours = float(step_hours)
        self.min_points = int(min_points)
        self.detection_quantum = None if detection_quantum is None else float(detection_quantum)
        self._pool_segments: List[Tuple[np.ndarray, np.ndarray]] = []
        self._pool_episodes: List[ShutInEpisode] = []
        self._app_inj: Optional[np.ndarray] = None
        self._app_seis: Optional[np.ndarray] = None
        self._app_episodes: Optional[List[ShutInEpisode]] = None
        self._run_start: Optional[np.ndarray] = None
        self._last_shape: Optional[DecayShape] = None
        self._pool_min_pos: float = np.inf
        self._app_minpos: Optional[np.ndarray] = None

    def add_pool_segment(self, inj_raw, seis_raw) -> None:
        inj = np.asarray(inj_raw, float)
        seis = np.asarray(seis_raw, float)
        self._pool_segments.append((inj, seis))
        self._pool_episodes.extend(
            extract_shutin_episodes(inj, seis, self.threshold_bbl, self.step_hours)
        )
        pos = seis[seis > 0]
        if len(pos):
            self._pool_min_pos = min(self._pool_min_pos, float(pos.min()))

    def set_application_series(self, inj_raw, seis_raw, zero_index: int) -> None:
        self._app_inj = np.asarray(inj_raw, float)
        self._app_seis = np.asarray(seis_raw, float)
        self.zero_index = int(zero_index)
        self._last_shape = None
        self._app_episodes = extract_shutin_episodes(
            self._app_inj, self._app_seis, self.threshold_bbl, self.step_hours
        )
        run_start = np.full(len(self._app_inj), -1, dtype=int)
        for ep in self._app_episodes:
            run_start[ep.start:ep.end] = ep.start
        self._run_start = run_start
        mp = np.full(len(self._app_seis) + 1, np.inf)
        run = np.inf
        for i, x in enumerate(self._app_seis):
            mp[i] = run
            if x > 0:
                run = min(run, float(x))
        mp[len(self._app_seis)] = run
        self._app_minpos = mp

    def _pool_points(self, origin: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dts, vals, sigs = [], [], []

        def _push(dt, v):
            if len(dt):
                dts.append(dt)
                vals.append(v)
                sigs.append(np.full(len(dt), np.sqrt(float(len(dt)))))

        for ep in self._pool_episodes:
            dt, v = ep.normalized_points(self.step_hours)
            _push(dt, v)
        for ep in self._app_episodes or []:
            if ep.start >= origin:
                continue
            cut = min(ep.end, origin) - ep.start
            if cut <= 0:
                continue
            observed = ep.rates[:cut]
            sub = ShutInEpisode(ep.start, ep.start + cut, observed,
                                _robust_start_level(observed))
            dt, v = sub.normalized_points(self.step_hours)
            _push(dt, v)
        if not dts:
            return np.empty(0), np.empty(0), np.empty(0)
        return np.concatenate(dts), np.concatenate(vals), np.concatenate(sigs)

    def _pool_residual_scale(self, shape: DecayShape, origin: int) -> Optional[float]:
        rs = []
        candidates = list(self._pool_episodes)
        for ep in self._app_episodes or []:
            if ep.end <= origin:
                candidates.append(ep)
        for ep in candidates:
            dt, v = ep.normalized_points(self.step_hours)
            if len(dt) < 3:
                continue
            ge = shape(dt)
            den = float(np.dot(ge, ge))
            if den <= _EPS:
                continue
            a = float(np.dot(ge, v) / den)
            rs.append(float(np.sqrt(np.mean((v - a * ge) ** 2))))
        return float(np.median(rs)) if rs else None

    def fit_windows(self, origins: Sequence[int]) -> List[Tuple[int, "WindowCorrection"]]:
        order = np.argsort(np.asarray(origins, dtype=float), kind="stable")
        fitted: List[Tuple[int, WindowCorrection]] = []
        for idx in order:
            fitted.append((int(origins[idx]), self.fit_window(int(origins[idx]))))
        inverse = np.empty(len(order), dtype=int)
        inverse[order] = np.arange(len(order))
        return [fitted[inverse[i]] for i in range(len(origins))]

    def fit_window(self, origin: int) -> WindowCorrection:
        origin = int(origin)
        dt, v, sig = self._pool_points(origin)
        shape = fit_decay_shape(dt, v, min_points=self.min_points)
        if shape.kind == "identity" and self._last_shape is not None:
            shape = self._last_shape
        elif shape.kind != "identity":
            self._last_shape = shape
        if shape.kind != "identity":
            g1 = float(shape(self.step_hours)[0])
            if g1 > _EPS:
                shape = DecayShape(shape.kind, shape.params, scale=shape.scale / g1)
                self._last_shape = shape

        wc = WindowCorrection(g=shape, a_ep=None, dt0_hours=0.0)
        if self._app_inj is None or not (0 <= origin < len(self._app_inj)):
            return wc

        s = self._run_start[origin] if self._run_start is not None else -1
        if s >= 0 and s < origin:
            ep = next(e for e in self._app_episodes if e.start == s)
            observed = ep.rates[: origin - s]
            dt_obs = (np.arange(1, len(observed) + 1)) * self.step_hours
            recent = observed[-80:]
            g_obs = shape(dt_obs[-len(recent):]) if len(recent) else np.empty(0)
            denom = float(np.dot(g_obs, g_obs))
            if denom > _EPS:
                wc.a_ep = float(np.dot(g_obs, recent) / denom)
            else:
                wc.a_ep = float(np.mean(observed)) if len(observed) else None
            wc.dt0_hours = (origin - s) * self.step_hours
            if wc.a_ep is not None:
                wc.a_ep = max(0.0, wc.a_ep)
                self._fill_v2_fields(wc, observed, dt_obs, shape, origin)
        return wc

    def _fill_v2_fields(
        self,
        wc: WindowCorrection,
        observed: np.ndarray,
        dt_obs: np.ndarray,
        shape: DecayShape,
        origin: int,
    ) -> None:
        wc.quiet = bool(np.any(observed == 0.0))
        if self.detection_quantum is not None:
            quantum = self.detection_quantum
        else:
            quantum = min(self._pool_min_pos, float(self._app_minpos[origin]))
        wc.floor_raw = 0.5 * quantum if np.isfinite(quantum) else 0.0

        dt_val = len(observed) * self.step_hours
        zi = getattr(self, "zero_index", None)
        if zi is not None:
            for ep in self._app_episodes or []:
                if ep.start >= zi and ep.end <= origin:
                    dt_val = max(dt_val, (ep.end - ep.start) * self.step_hours)
        wc.dt_val_hours = dt_val

        level = _robust_start_level(observed)
        if level <= 0:
            wc.misfit, wc.w = 0.0, 1.0
            return
        if len(observed) < 3 or shape.kind == "identity":
            wc.w = 0.0
            return
        tail = observed[-80:]
        resid = tail - wc.a_ep * shape(dt_obs[-len(tail):])
        wc.misfit = float(np.sqrt(np.mean(resid ** 2)) / level)
        wc.pool_scale = self._pool_residual_scale(shape, origin)
        if wc.pool_scale is None:
            wc.w = 0.0
            return
        s = max(wc.pool_scale, 1e-6)
        wc.w = float(min(1.0, (s / max(wc.misfit, 1e-12)) ** 2))

    def run_start_of(self, idx: int) -> int:
        if self._run_start is None or not (0 <= idx < len(self._run_start)):
            return -1
        return int(self._run_start[idx])

    @property
    def app_injection(self) -> np.ndarray:
        if self._app_inj is None:
            raise RuntimeError("set_application_series 未调用")
        return self._app_inj

    @property
    def app_seismic(self) -> np.ndarray:
        if self._app_seis is None:
            raise RuntimeError("set_application_series 未调用")
        return self._app_seis


def apply_correction(
    pred: np.ndarray,
    window_corrections: List[Tuple[int, WindowCorrection]],
    estimator: ShutInDecayEstimator,
    seis_range: float,
    future_ops: bool = True,
) -> np.ndarray:
    out = np.array(pred, copy=True)
    inj = estimator.app_injection
    thr = estimator.threshold_bbl
    step = estimator.step_hours
    n_dec = pred.shape[1]
    for w, (origin, wc) in enumerate(window_corrections):
        if not future_ops and wc.a_ep is None:
            continue
        for t in range(n_dec):
            i = origin + t
            if future_ops:
                if i >= len(inj) or inj[i] > thr:
                    continue
                s = estimator.run_start_of(i)
                if s < 0 or s >= origin:
                    continue
                if wc.a_ep is None:
                    continue
                dt = (i - s + 1) * step
            else:
                dt = wc.dt0_hours + (t + 1) * step
            g_dt = float(wc.g(np.array([dt]))[0])
            if dt > wc.dt_val_hours:
                g_edge = float(wc.g(np.array([wc.dt_val_hours]))[0])
                gamma = wc.dt_val_hours / dt
                g_dt = gamma * g_dt + (1.0 - gamma) * g_edge
            curve_z = (wc.a_ep / seis_range) * g_dt
            if wc.quiet and curve_z < wc.floor_raw / seis_range:
                out[w, t] = 0.0
                continue
            if wc.w <= 0.0:
                continue
            qbar = float(np.mean(pred[w, t]))
            target_z = wc.w * curve_z + (1.0 - wc.w) * qbar
            if abs(qbar) > _EPS:
                out[w, t] = pred[w, t] * (target_z / qbar)
            else:
                out[w, t] = target_z
            np.clip(out[w, t], 0.0, None, out=out[w, t])
    return out


class ShutInDecayTFT(TemporalFusionTransformer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutin_estimator: Optional[ShutInDecayEstimator] = None
        self._primed_windows: Optional[List[Tuple[int, WindowCorrection]]] = None
        self._seis_range: Optional[float] = None
        self._future_ops: bool = True
        self._alignment_checked = False

    def prime(
        self,
        estimator: ShutInDecayEstimator,
        window_origins: Sequence[int],
        seis_range: float,
        future_ops: bool = True,
    ) -> "ShutInDecayTFT":
        self.shutin_estimator = estimator
        self._seis_range = float(seis_range)
        self._future_ops = bool(future_ops)
        self._primed_windows = estimator.fit_windows(window_origins)
        self._alignment_checked = False
        return self

    @property
    def primed_windows(self) -> Optional[List[Tuple[int, WindowCorrection]]]:
        return self._primed_windows

    def _verify_alignment(self, out) -> None:
        est = self.shutin_estimator
        seis = est.app_seismic
        targets = out.x["decoder_target"].detach().cpu().numpy()
        n_dec = targets.shape[1]
        origins = np.array([o for o, _ in self._primed_windows])
        obs = np.stack([seis[o:o + n_dec] / self._seis_range for o in origins])
        if obs.shape != targets.shape or not np.allclose(targets, obs, rtol=1e-3, atol=1e-4):
            raise RuntimeError(
                "ShutInDecayTFT 对齐校验失败: "
                "predict 输出顺序与 prime 的 window_origins 不一致"
            )
        self._alignment_checked = True

    def predict(self, dataloader, mode="raw", return_x=True, **kwargs):
        out = super().predict(dataloader, mode=mode, return_x=return_x, **kwargs)
        if self._primed_windows is None or mode != "raw":
            warnings.warn("ShutInDecayTFT: 未 prime 或 mode != 'raw', 返回原 TFT 预测")
            return out
        pred = out.output["prediction"]
        if isinstance(pred, (list, tuple)) or pred.dim() != 3:
            warnings.warn("ShutInDecayTFT: 非单目标 3D 预测, 跳过修正")
            return out
        if pred.shape[0] != len(self._primed_windows):
            warnings.warn(
                f"ShutInDecayTFT: 预测窗口数 ({pred.shape[0]}) 与 prime 窗口数 "
                f"({len(self._primed_windows)}) 不一致, 跳过修正"
            )
            return out
        if not self._alignment_checked:
            self._verify_alignment(out)
        corrected = apply_correction(
            pred.detach().cpu().numpy(),
            self._primed_windows,
            self.shutin_estimator,
            self._seis_range,
            future_ops=self._future_ops,
        )
        new_tensor = torch.as_tensor(corrected, dtype=pred.dtype, device=pred.device)
        return out._replace(output=out.output._replace(prediction=new_tensor))

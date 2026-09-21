import os
import csv
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.io
import h5py
import json
import torch
from obspy import UTCDateTime
from math import radians, degrees, sin, cos, sqrt, asin

from scipy.ndimage import gaussian_filter, gaussian_filter1d

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def quantile_calibration(y, q_dict=None, alphas=None, Q=None):
    y = np.asarray(y).ravel()
    T = len(y)

    if q_dict is not None:
        alphas_sorted = np.array(sorted(q_dict.keys()))
        Qmat = np.stack([np.asarray(q_dict[a]).ravel() for a in alphas_sorted], axis=1)
    else:
        assert Q is not None and alphas is not None, "Provide q_dict or (Q, alphas)"
        alphas_sorted = np.asarray(alphas).ravel()
        sort_idx = np.argsort(alphas_sorted)
        alphas_sorted = alphas_sorted[sort_idx]
        Qmat = np.asarray(Q)
        Qmat = Qmat[:, sort_idx]

    emp = (y[:, None] <= Qmat).mean(axis=0)

    ks = np.max(np.abs(emp - alphas_sorted))
    return alphas_sorted, emp, ks


def add_lr_forecast(data, pred_len, reals):
    seismic_rate = data['seismic_rate']
    external_drivers = []
    for key in reals:
        external_drivers.append(data[key])

    lr_result = lr_seismic_forecast(
        seismic_rate,
        external_drivers,
        forecast=pred_len,     
    )
    data_dict = dict(data)
    seismic_rate_lr = np.concatenate([seismic_rate[:-pred_len], lr_result])
    data_dict['seismic_rate_lr'] = seismic_rate_lr
    return data_dict

def _as_inj_mat(inj):
    if isinstance(inj, (list, tuple)) and len(inj) > 0 and not hasattr(inj, "shape"):
        cols = [np.asarray(col, dtype=float).ravel() for col in inj]
        return np.column_stack(cols)
    arr = np.asarray(inj, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def run_arx_forecast(seis, inj, forecast):

    nl = len(seis)
    all_index = np.arange(nl)
    target_index = all_index[-forecast:]
    input_index = all_index[:-forecast] if forecast > 0 else all_index.copy()

    if isinstance(inj, (list, tuple)) and len(inj) > 0 and not hasattr(inj, "shape"):
        cols = [np.asarray(col, dtype=float).ravel() for col in inj]
        inj_mat = np.column_stack(cols)
    else:
        inj_mat = inj.reshape(-1, 1)
    F = inj_mat.shape[1]

    if len(input_index) < 2:
        raise ValueError("Need at least two known timesteps to build lagged features.")

    X_train, y_train = [], []
    for k in range(1, len(input_index)):
        idx = input_index[k]
        prev_idx = input_index[k - 1]
        x_feat = np.concatenate([inj_mat[idx], [seis[prev_idx]]])  
        X_train.append(x_feat)
        y_train.append(seis[idx])                 
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = []
    prev_seis = seis[input_index[-1]]
    for t in target_index:
        x_feat = np.concatenate([inj_mat[t], [prev_seis]]).reshape(1, F + 1)
        yhat = model.predict(x_feat)[0]
        preds.append(yhat)
        prev_seis = yhat
    preds = np.asarray(preds)

    return preds

def run_gbdt_forecast(seis, 
                      inj, 
                      forecast,
                      n_estimators=100, 
                      max_depth=3, 
                      learning_rate=0.1):

    nl = len(seis)
    all_index = np.arange(nl)
    target_index = all_index[-forecast:]
    input_index = all_index[:-forecast] if forecast > 0 else all_index.copy()

    if isinstance(inj, (list, tuple)) and len(inj) > 0 and not hasattr(inj, "shape"):
        cols = [np.asarray(col, dtype=float).ravel() for col in inj]
        inj_mat = np.column_stack(cols)
    else:
        inj_mat = inj.reshape(-1, 1)
    F = inj_mat.shape[1]

    if len(input_index) < 2:
        raise ValueError("Need at least two known timesteps to build lagged features.")

    X_train, y_train = [], []
    for k in range(1, len(input_index)): 
        idx = input_index[k]
        prev_idx = input_index[k - 1]
        x_feat = np.concatenate([inj_mat[idx], [seis[prev_idx]]])  
        y_train.append(seis[idx])             
        X_train.append(x_feat)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model_gbdt = xgb.XGBRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        learning_rate=learning_rate
    )
    model_gbdt.fit(X_train, y_train)

    gbdt_preds = []
    prev_seis = seis[input_index][-1] 
    for t in target_index:

        x_feat = np.concatenate([inj_mat[t], [prev_seis]]).reshape(1, F + 1)
        yhat = model_gbdt.predict(x_feat)[0]

        gbdt_preds.append(yhat)
        prev_seis = yhat

    gbdt_preds = np.array(gbdt_preds)

    return gbdt_preds


def fit_arx_model(seis, inj):
    seis = np.asarray(seis, dtype=float)
    inj_mat = _as_inj_mat(inj)
    X, y = [], []
    for t in range(1, len(seis)):
        X.append(np.concatenate([inj_mat[t], [seis[t - 1]]]))
        y.append(seis[t])
    model = LinearRegression()
    model.fit(np.asarray(X), np.asarray(y))
    return model


def predict_arx_frozen(model, seis_last, inj_future, n_steps):
    inj_mat = _as_inj_mat(inj_future)
    assert len(inj_mat) >= n_steps
    F = inj_mat.shape[1]
    preds, prev = [], float(seis_last)
    for t in range(n_steps):
        yhat = model.predict(np.concatenate([inj_mat[t], [prev]]).reshape(1, F + 1))[0]
        preds.append(yhat)
        prev = yhat
    return np.asarray(preds)


def fit_gbdt_model(seis, inj, n_estimators=100, max_depth=3, learning_rate=0.1):
    seis = np.asarray(seis, dtype=float)
    inj_mat = _as_inj_mat(inj)
    X, y = [], []
    for t in range(1, len(seis)):
        X.append(np.concatenate([inj_mat[t], [seis[t - 1]]]))
        y.append(seis[t])
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    model.fit(np.asarray(X), np.asarray(y))
    return model


def predict_gbdt_frozen(model, seis_last, inj_future, n_steps):
    inj_mat = _as_inj_mat(inj_future)
    assert len(inj_mat) >= n_steps
    F = inj_mat.shape[1]
    preds, prev = [], float(seis_last)
    for t in range(n_steps):
        yhat = model.predict(np.concatenate([inj_mat[t], [prev]]).reshape(1, F + 1))[0]
        preds.append(yhat)
        prev = yhat
    return np.asarray(preds)

def evaluate_time_series_similarity(
    a, bs, verbose=True,
    reduction="mean",
    dtw_cost="L1",
    dtw_warp_penalty=0.0,
    cosine_mode="one_minus",
    pearson_mode="one_minus_r",
    add_relative_metrics=True,
    quantiles=None,
):
    bs_arr = np.asarray(bs, dtype=float)
    if bs_arr.ndim == 1:
        bs_arr = bs_arr[:, None]
    b = bs_arr.mean(axis=1)

    xa = np.asarray(a, dtype=float).ravel()
    xb = np.asarray(b, dtype=float).ravel()
    if xa.size == 0 or xb.size == 0:
        raise ValueError("Input time series must be non-empty.")

    n = min(xa.size, xb.size)
    xa, xb = xa[:n], xb[:n]

    mask = np.isfinite(xa) & np.isfinite(xb)
    if not np.any(mask):
        raise ValueError("No overlapping finite values after NaN filtering.")
    xa, xb = xa[mask], xb[mask]

    diff = xa - xb
    if reduction == "sum":
        sse = float(np.sum(diff**2))
        mse = sse / len(diff)       
        rmse = float(np.sqrt(mse))
        mae = float(np.sum(np.abs(diff)) / len(diff))
    else: 
        mse  = float(np.mean(diff**2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(diff)))

    def dtw_distance(x, y):
        nx, ny = len(x), len(y)
        D = np.full((nx + 1, ny + 1), np.inf, dtype=float)
        D[0, 0] = 0.0
        for i in range(1, nx + 1):
            xi = x[i - 1]
            for j in range(1, ny + 1):
                delta = xi - y[j - 1]
                if dtw_cost.upper() == "L1":
                    cost = abs(delta)
                elif dtw_cost.upper() == "L2":
                    cost = delta ** 2
                else:
                    raise ValueError("dtw_cost must be 'L1' or 'L2'")
                insertion = D[i - 1, j] + dtw_warp_penalty
                deletion = D[i, j - 1] + dtw_warp_penalty
                match = D[i - 1, j - 1]
                D[i, j] = cost + min(insertion, deletion, match)
        distance = D[nx, ny]
        if dtw_cost.upper() == "L2":
            distance = np.sqrt(distance)
        return float(distance)
    dtw = dtw_distance(xa, xb)

    su, sv = np.std(xa), np.std(xb)
    if su == 0 or sv == 0:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(xa, xb)[0, 1])

    if pearson_mode == "one_minus_r2":
        pearson_dist = 1.0 - (pearson_r ** 2)   
    else:
        pearson_dist = 1.0 - pearson_r   

    nu, nv = np.linalg.norm(xa), np.linalg.norm(xb)
    if nu == 0 or nv == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = float(np.dot(xa, xb) / (nu * nv))

    if cosine_mode == "angle":
        cosine_distance = float(np.arccos(np.clip(cosine_sim, -1.0, 1.0)))
    else:
        cosine_distance = 1.0 - cosine_sim

    results = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "DTW_distance": dtw,
        "Pearson_correlation": pearson_r,
        "Pearson_distance": float(pearson_dist),
        "Cosine_similarity": cosine_sim,
        "Cosine_distance": float(cosine_distance),
    }

    if add_relative_metrics:
        eps = 1e-12
        mask_rel = np.abs(xa) > eps
        if np.any(mask_rel):
            mape = float(np.mean(np.abs((xa[mask_rel] - xb[mask_rel]) / xa[mask_rel])) * 100.0)
        else:
            mape = float("nan")
        smape = float(np.mean(2.0 * np.abs(xa - xb) / (np.abs(xa) + np.abs(xb) + eps)) * 100.0)
        results.update({"MAPE_%": mape, "SMAPE_%": smape})

    if quantiles is not None:
        quantiles = [float(q) for q in quantiles]
        if bs_arr.shape[1] != len(quantiles):
            raise ValueError(
                f"bs has {bs_arr.shape[1]} columns, but quantiles contains {len(quantiles)} values"
            )
        qp = bs_arr[:n][mask]
        y = xa.copy()
        finite = np.all(np.isfinite(qp), axis=1) & np.isfinite(y)
        qp, y = qp[finite], y[finite]
        if qp.shape[0] == 0:
            raise ValueError("bs and target have no valid overlapping rows")
        for k, q in enumerate(quantiles):
            pred_q = qp[:, k]
            err = y - pred_q
            pinball = np.where(err >= 0, q * err, (q - 1.0) * err)
            results[f"pinball_q{q:g}"] = float(np.mean(pinball))
            results[f"coverage_q{q:g}"] = float(np.mean(pred_q >= y))
        results["pinball_mean"] = float(np.mean([results[f"pinball_q{q:g}"] for q in quantiles]))
        lo, hi = quantiles[0], quantiles[-1]
        results[f"interval_cov_[{lo:g},{hi:g}]"] = float(np.mean((y >= qp[:, 0]) & (y <= qp[:, -1])))
        results[f"interval_tgt_[{lo:g},{hi:g}]"] = hi - lo

    if verbose:
        print("Time Series Similarity Metrics (configurable)")
        print("-------------------------------------------")
        base_keys = ["MSE", "RMSE", "MAE", "DTW_distance",
                     "Pearson_correlation", "Pearson_distance",
                     "Cosine_similarity", "Cosine_distance"]
        if add_relative_metrics:
            base_keys += ["MAPE_%", "SMAPE_%"]
        for k in base_keys:
            if k in results:
                print(f"{k:21s}: {results[k]:.6g}")
        if quantiles is not None:
            print("--- quantile forecast (pinball / coverage vs ideal) ---")
            print(f"{'quantile':>10s} {'pinball':>14s} {'coverage':>12s} {'ideal':>10s}")
            for q in quantiles:
                print(f"{q:10.4g} {results[f'pinball_q{q:g}']:14.6g} "
                      f"{results[f'coverage_q{q:g}']:12.6g} {q:10.4g}")
            print(f"{'mean':>10s} {results['pinball_mean']:14.6g}")
            lo, hi = quantiles[0], quantiles[-1]
            print(f"interval [{lo:g},{hi:g}]: coverage={results[f'interval_cov_[{lo:g},{hi:g}]']:.4f}, "
                  f"ideal={results[f'interval_tgt_[{lo:g},{hi:g}]']:.4f}")

    return results

def merge_quantile_segments(pred_list, index_list, agg="mean"):

    assert len(pred_list) == len(index_list) and len(pred_list) > 0

    if len(pred_list[0].shape) == 1:
        pred_list = [x[:, None] for x in pred_list]

    Q = pred_list[0].shape[1]

    for p, idx in zip(pred_list, index_list):
        assert p.ndim == 2 and p.shape[1] == Q
        assert len(idx) == p.shape[0]

    all_idx = np.concatenate([np.asarray(idx).ravel() for idx in index_list])
    i_min, i_max = int(all_idx.min()), int(all_idx.max())
    global_index = np.arange(i_min, i_max + 1, dtype=int)
    T_global = len(global_index)
    K = len(pred_list)

    stack = np.full((T_global, Q, K), np.nan, dtype=float)
    for k, (pred, idx) in enumerate(zip(pred_list, index_list)):
        idx = np.asarray(idx, dtype=int)
        pos = idx - i_min                   
        stack[pos, :, k] = pred

    if agg == "mean":
        merged = np.nanmean(stack, axis=2)
    elif agg == "median":
        merged = np.nanmedian(stack, axis=2)
    elif agg == "last":
        merged = np.empty((T_global, Q), dtype=float)
        merged[:] = np.nan
        for k in range(K-1, -1, -1):
            mask = np.isnan(merged) & ~np.isnan(stack[:, :, k])
            merged[mask] = stack[:, :, k][mask]
    elif agg == "first":
        merged = np.empty((T_global, Q), dtype=float)
        merged[:] = np.nan
        for k in range(K):
            mask = np.isnan(merged) & ~np.isnan(stack[:, :, k])
            merged[mask] = stack[:, :, k][mask]
    else:
        raise ValueError("agg must be one of: mean | median | last | first")

    return merged, global_index


def downsample_time_series(time_array, *float_arrays, rule="M", agg="mean"):
    data = {"time": time_array}
    col_names = [] 

    for i, arr in enumerate(float_arrays, 1):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            col = f"val{i}"
            data[col] = arr
            col_names.append([col])
        elif arr.ndim == 2:  
            cols = [f"val{i}_{j}" for j in range(arr.shape[1])]
            for j, c in enumerate(cols):
                data[c] = arr[:, j]
            col_names.append(cols)
        else:
            raise ValueError("only support 1D or 2D array")

    df = pd.DataFrame(data).set_index("time")
    df_resampled = getattr(df.resample(rule), agg)()

    time_down = df_resampled.index.to_pydatetime()
    results = [time_down]

    for cols in col_names:
        arr_down = df_resampled[cols].to_numpy()
        if len(cols) == 1:
            arr_down = arr_down.ravel()
        results.append(arr_down)
    return tuple(results)

def lists_to_df(list_of_dicts, target, known_reals, explode_cols=['time_idx', 'date']):

    sorted_list = []
    for x in list_of_dicts:
        order = np.argsort(x["time_idx"])
        sorted_dict = {k: (v[order] if isinstance(v, np.ndarray) and v.shape[0] == len(order) else v)
                       for k, v in x.items()}
        sorted_list.append(sorted_dict)

    if isinstance(target, list):
        target_names = target
    else:
        if len(sorted_list[0][target].shape) > 1:
            n_channels = sorted_list[0][target].shape[1]
            target_names = [f"{target}_{i}" for i in range(n_channels)]
            for x in sorted_list:
                for i, name in enumerate(target_names):
                    x[name] = x[target][:, i]
                del x[target]
        else:
            target_names = [target]

    df = pd.DataFrame(sorted_list)

    explode_cols = explode_cols + target_names + known_reals
    if "mask" in df.columns:
        explode_cols.append("mask")
    df = df.explode(explode_cols, ignore_index=True)

    df['time_idx'] = df['time_idx'].astype('int64')
    df['group_id'] = df['group_id'].astype('int64')
    for col in target_names + known_reals:
        df[col] = pd.to_numeric(df[col])
    if "mask" in df.columns:
        df["mask"] = pd.to_numeric(df["mask"])

    return df

def decay_to_zero(time_series_in, start_index, mode='linear', decay_fraction=1.0):
    import copy
    time_series = copy.deepcopy(time_series_in)
    if start_index < 0 or start_index >= len(time_series):
        raise ValueError("start_index is outside the time series")

    n = len(time_series) - start_index
    if n == 0:
        return time_series 

    decay_steps = int(n * decay_fraction)

    if mode == 'linear':
        decay_weights = np.linspace(1, 0, decay_steps)
    elif mode == 'exponential':
        decay_weights = np.exp(-np.linspace(0, 5, decay_steps))
        decay_weights = (decay_weighcompute_b_valuets - decay_weights.min()) / (decay_weights.max() - decay_weights.min())
    else:
        raise ValueError("mode must be 'linear' or 'exponential'")

    full_decay_weights = np.zeros(n)
    full_decay_weights[:decay_steps] = decay_weights

    time_series[start_index:] *= full_decay_weights

    return time_series

def interpret_output(
    out,
    max_encoder_length,
):
    from pytorch_forecasting.utils import create_mask, masked_op
    batch_size = len(out["decoder_attention"])
    if isinstance(out["decoder_attention"], (list, tuple)):
        max_last_dimension = max(x.size(-1) for x in out["decoder_attention"])
        first_elm = out["decoder_attention"][0]
        decoder_attention = torch.full(
            (batch_size, *first_elm.shape[:-1], max_last_dimension),
            float("nan"),
            dtype=first_elm.dtype,
            device=first_elm.device,
        )
        for idx, x in enumerate(out["decoder_attention"]):
            decoder_length = out["decoder_lengths"][idx]
            decoder_attention[idx, :, :, :decoder_length] = x[..., :decoder_length]
    else:
        decoder_attention = out["decoder_attention"].clone()
        decoder_mask = create_mask(out["decoder_attention"].size(1), out["decoder_lengths"])
        decoder_attention[decoder_mask[..., None, None].expand_as(decoder_attention)] = float("nan")

    if isinstance(out["encoder_attention"], (list, tuple)):
        first_elm = out["encoder_attention"][0]
        encoder_attention = torch.full(
            (batch_size, *first_elm.shape[:-1], max_encoder_length),
            float("nan"),
            dtype=first_elm.dtype,
            device=first_elm.device,
        )
        for idx, x in enumerate(out["encoder_attention"]):
            encoder_length = out["encoder_lengths"][idx]
            encoder_attention[idx, :, :, max_encoder_length - encoder_length :] = x[
                ..., :encoder_length
            ]
    else:
        encoder_attention = out["encoder_attention"].clone()
        shifts = encoder_attention.size(3) - out["encoder_lengths"]
        new_index = (
            torch.arange(encoder_attention.size(3), device=encoder_attention.device)[None, None, None].expand_as(
                encoder_attention
            )
            - shifts[:, None, None, None]
        ) % encoder_attention.size(3)
        encoder_attention = torch.gather(encoder_attention, dim=3, index=new_index)
        if encoder_attention.size(-1) < max_encoder_length:
            encoder_attention = torch.concat(
                [
                    torch.full(
                        (
                            *encoder_attention.shape[:-1],
                            max_encoder_length - out["encoder_lengths"].max(),
                        ),
                        float("nan"),
                        dtype=encoder_attention.dtype,
                        device=encoder_attention.device,
                    ),
                    encoder_attention,
                ],
                dim=-1,
            )

    attention = torch.concat([encoder_attention, decoder_attention], dim=-1)
    attention[attention < 1e-5] = float("nan")
    return attention

def find_segments(arr, threshold):
    mask = arr > threshold
    diff = np.diff(mask.astype(int))

    start_indices = np.where(diff == 1)[0] + 1
    end_indices = np.where(diff == -1)[0] + 1

    if mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if mask[-1]:
        end_indices = np.append(end_indices, len(arr))

    return list(zip(start_indices, end_indices))


def masked_op(array: np.ndarray, op: str = "mean", axis: int = 0, mask: np.ndarray = None) -> np.ndarray:
    if mask is None:
        mask = ~np.isnan(array)

    masked_array = np.where(mask, array, 0.0)
    summed = np.sum(masked_array, axis=axis)

    if op == "mean":
        count = np.sum(mask, axis=axis)
        return np.divide(summed, count, where=count > 0)
    elif op == "sum":
        return summed
    else:
        raise ValueError(f"Unknown operation {op}")

def decay_to_zero(time_series, start_index, mode='linear', decay_fraction=1.0):
    if start_index < 0 or start_index >= len(time_series):
        raise ValueError("start_index is outside the time series")

    n = len(time_series) - start_index
    if n == 0:
        return time_series

    decay_steps = int(n * decay_fraction)

    if mode == 'linear':
        decay_weights = np.linspace(1, 0, decay_steps)
    elif mode == 'exponential':
        decay_weights = np.exp(-np.linspace(0, 5, decay_steps))
        decay_weights = (decay_weights - decay_weights.min()) / (decay_weights.max() - decay_weights.min())
    else:
        raise ValueError("mode must be 'linear' or 'exponential'")

    full_decay_weights = np.zeros(n)
    full_decay_weights[:decay_steps] = decay_weights

    time_series[start_index:] *= full_decay_weights

    return time_series

def smooth_with_fixed_nonzero(data, sigma=12):
    mask = data != 0
    result = np.zeros_like(data)
    for idx in np.where(mask)[0]:
        left = max(0, idx - 3 * sigma)
        right = min(len(data), idx + 3 * sigma + 1)
        local_data = np.zeros_like(data)
        local_data[idx] = data[idx] 
        smoothed_local = gaussian_filter1d(local_data, sigma=sigma)[left:right]
        smoothed_local = min_max_norm(smoothed_local) * data[idx]
        result[left:right] = np.maximum(result[left:right], smoothed_local)
    return result

def is_point_in_triangle(point, triangle):
    x, y = point
    x1, y1 = triangle[0]
    x2, y2 = triangle[1]
    x3, y3 = triangle[2]
    detT = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if detT == 0:
        return False
    alpha = ((x2 - x) * (y3 - y) - (x3 - x) * (y2 - y)) / detT
    beta = ((x3 - x) * (y1 - y) - (x1 - x) * (y3 - y)) / detT
    gamma = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / detT
    return alpha >= 0 and beta >= 0 and gamma >= 0 and (alpha + beta + gamma <= 1)

def get_middle_element(lst):
    n = len(lst)
    if n == 0:
        return None  
    middle_index = n // 2
    if n % 2 == 1:  
        return lst[middle_index]
    else: 
        return lst[middle_index - 1] 

def min_max_norm(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)

def mea_std_norm(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    return (x - x_mean) / x_std

def bounding_box(lat, lon, distance):
    R = 6371
    lat_radians = radians(lat)
    lon_radians = radians(lon)
    delta_lat = distance / R
    min_lat = lat - degrees(delta_lat)
    max_lat = lat + degrees(delta_lat)
    delta_lon = distance / (R * cos(lat_radians))
    min_lon = lon - degrees(delta_lon)
    max_lon = lon + degrees(delta_lon)
    return (min_lat, max_lat, min_lon, max_lon)

def sequence_roll(sequence, shift, mode='constant'):
    sequence_arr = np.array(sequence, dtype=np.single)
    if shift == 0:
        return sequence_arr 
    shift_abs = abs(shift)
    if mode == 'constant':
        sequence_pad = np.pad(sequence_arr, ((shift_abs, shift_abs)), mode='constant', constant_values=0)
    else:
        sequence_pad = np.pad(sequence_arr, ((shift_abs, shift_abs)), mode=mode)
    sequence_roll = np.roll(sequence_pad, shift)
    return sequence_roll[shift_abs:-shift_abs]

def date2timestamp(date):
    timestamp = pd.Timestamp(date, tz='UTC')
    unix_timestamp_seconds = int(timestamp.timestamp())
    unix_timestamp_days = unix_timestamp_seconds // (24 * 3600)
    return unix_timestamp_days

def timestamp2date(timestamp):
    date = pd.to_datetime(timestamp * 24 * 3600, unit='s', utc=True)
    return date

def nan2num(array):
    series = pd.Series(array)
    return series.interpolate(method='linear').fillna(0.).to_numpy()

def write_csv_from_dict(path, data):
    if len(data) > 0:
        fieldnames = data[0].keys()

        with open(path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()

            for row in data:
                writer.writerow(row)

def write_json(path, x, indent=4):
    with open(path, 'w+', encoding='utf8') as fp:
        fp.write(json.dumps(x, indent=indent))

def read_json(path):
    with open(path, 'r+', encoding='utf8') as fp:
        json_data = json.load(fp)
    return json_data

def read_csv(path):
    data = []
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def write_csv(path, data):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

def read_text(path):
    text = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                text.append(line)
    return text

def read_hdpy(path):
    dataset = dict()
    with h5py.File(path, mode='r') as in_file:
        for key in in_file.keys():
            dataset[key] = in_file[key][()]
    return dataset

def write_hdpy(path, dataset):
    with h5py.File(path, mode='w') as out_file:
        for key in dataset.keys():
            out_file.create_dataset(
                key,
                data=dataset[key]
            ) 

def get_date_time(ordinal_date_with_fraction):
    integer_part = int(ordinal_date_with_fraction)
    fractional_part = ordinal_date_with_fraction - integer_part

    gregorian_date_with_fraction = datetime.fromordinal(integer_part)

    seconds_in_day = 24 * 60 * 60
    additional_seconds = fractional_part * seconds_in_day

    final_date_time = gregorian_date_with_fraction + timedelta(seconds=additional_seconds)
    return final_date_time

def date_time_to_ordinal_with_fraction(date_time):
    ordinal_date = date_time.toordinal()

    seconds_since_midnight = (date_time - datetime(date_time.year, date_time.month, date_time.day)).total_seconds()

    seconds_in_day = 24 * 60 * 60

    fractional_day = seconds_since_midnight / seconds_in_day

    ordinal_date_with_fraction = ordinal_date + fractional_day
    return ordinal_date_with_fraction

def merge_list_of_dicts(dict_list, target):
    merged = {}
    for key in dict_list[0].keys():
        merged[key] = np.concatenate([d[key] for d in dict_list], axis=0)

    n_channels = merged[target].shape[1]
    target_names = [f"{target}_{i}" for i in range(n_channels)]
    for i, name in enumerate(target_names):
        merged[name] = merged[target][:, i]
    del merged[target]
    return merged, target_names


def normalize_features(
    seismic_rate,
    injection_rate,
    injection_rate_grad,
    pressure=None,
    pressure_grad=None,
    normm="mea_std",
):

    if normm not in ["mea_std", "min_max"]:
        raise ValueError(
            f"Unsupported normalization method: normm={normm!r}. "
            f"Choose from 'mea_std' or 'min_max'."
        )

    if normm == "mea_std":
        seismic_rate_mean = seismic_rate.mean()
        seismic_rate_std = seismic_rate.std()
        seismic_rate_norm = mea_std_norm(seismic_rate)

        injection_rate_mean = injection_rate.mean()
        injection_rate_std = injection_rate.std()
        injection_rate_norm = mea_std_norm(injection_rate)

        injection_rate_grad_mean = injection_rate_grad.mean()
        injection_rate_grad_std = injection_rate_grad.std()
        injection_rate_grad_norm = mea_std_norm(injection_rate_grad)

        params = {
            "seismic_rate": {"mean": seismic_rate_mean, "std": seismic_rate_std},
            "injection_rate": {"mean": injection_rate_mean, "std": injection_rate_std},
            "injection_rate_grad": {
                "mean": injection_rate_grad_mean, 
                "std": injection_rate_grad_std,
            }
        }

        output = [seismic_rate_norm, injection_rate_norm, injection_rate_grad_norm] 
        if pressure is not None:
            pressure_mean = pressure.mean()
            pressure_std = pressure.std()            
            pressure_norm = mea_std_norm(pressure)
            params['pressure'] = {"mean": pressure_mean, "std": pressure_std}
            output.append(pressure_norm)

        if pressure_grad is not None:
            pressure_grad_mean = pressure_grad.mean()
            pressure_grad_std = pressure_grad.std()    
            pressure_grad_norm = mea_std_norm(pressure_grad)
            params['pressure_grad'] = {
                "mean": pressure_grad_mean, 
                "std": pressure_grad_std,
            }   
            output.append(pressure_grad_norm)


    elif normm == "min_max":
        seismic_rate_min = seismic_rate.min()
        seismic_rate_range = seismic_rate.max() - seismic_rate.min()
        seismic_rate_norm = min_max_norm(seismic_rate)

        injection_rate_min = injection_rate.min()
        injection_rate_range = injection_rate.max() - injection_rate.min()
        injection_rate_norm = min_max_norm(injection_rate)

        injection_rate_grad_min = injection_rate_grad.min()
        injection_rate_grad_range = injection_rate_grad.max() - injection_rate_grad.min()
        injection_rate_grad_norm = min_max_norm(injection_rate_grad)

        params = {
            "seismic_rate": {"min": seismic_rate_min, "range": seismic_rate_range},
            "injection_rate": {"min": injection_rate_min, "range": injection_rate_range},
            "injection_rate_grad": {
                "min": injection_rate_grad_min, 
                "range": injection_rate_grad_range
            }
        }

        output = [seismic_rate_norm, injection_rate_norm, injection_rate_grad_norm] 
        if pressure is not None:
            pressure_min = pressure.min()
            pressure_range = pressure.max() - pressure.min()
            pressure_norm = min_max_norm(pressure)
            params['pressure'] = {"min": pressure_min, "range": pressure_range}
            output.append(pressure_norm)

        if pressure_grad is not None:
            pressure_grad_min = pressure_grad.min()
            pressure_grad_range = pressure_grad.max() - pressure_grad.min()
            pressure_grad_norm = min_max_norm(pressure_grad)
            params['pressure_grad'] = {
                "min": pressure_grad_min, 
                "range": pressure_grad_range,
            }    
            output.append(pressure_grad_norm)

    else:
        raise ValueError(
            f"Unsupported normalization method: normm={normm!r}."
        )

    output.append(params)
    return output

def predict_and_collect_results(
    model, apply_dataloader, apply_data_set, application, time_axis, norm_stats,
):

    raw_predictions = model.predict(apply_dataloader, mode="raw", return_x=True)

    ad = apply_data_set.sort_values(['group_id', 'time_idx']).to_dict(orient='list')
    seismic_rate_all = np.array(ad['seismic_rate'])
    injection_rate_all = np.array(ad['injection_rate'])

    pressure_all = np.array(ad['pressure']) if 'pressure' in ad.keys() else None

    injection_rate_grad_all = np.array(ad['injection_rate_grad'])
    group_ids = np.array(ad['group_id'])
    date_all = np.array(ad['date'])

    target_all = raw_predictions.x['decoder_target'].detach().cpu().numpy()
    decoder_time_idx = raw_predictions.x['decoder_time_idx'].detach().cpu().numpy()
    prediction_all = raw_predictions.output['prediction'].detach().cpu().numpy()

    results = []
    for j in range(len(target_all)):
        index = application.index.iloc[j]
        s, e = index.index_start, index.index_end + 1
        this_date = date_all[s:e]
        this_group_id = group_ids[s:e]
        this_seismic_rate = seismic_rate_all[s:e] * norm_stats['seismic_rate'][1] + norm_stats['seismic_rate'][0]
        this_injection_rate = injection_rate_all[s:e] * norm_stats['injection_rate'][1] + norm_stats['injection_rate'][0]

        if pressure_all is not None:
            this_pressure = pressure_all[s:e] * norm_stats['pressure'][1] + norm_stats['pressure'][0]
        else:
            this_pressure = None

        this_injection_rate_grad = injection_rate_grad_all[s:e] * norm_stats['injection_rate_grad'][1] + norm_stats['injection_rate_grad'][0]
        this_target = target_all[j]  * norm_stats['seismic_rate'][1] + norm_stats['seismic_rate'][0]
        this_prediction = prediction_all[j]  * norm_stats['seismic_rate'][1] + norm_stats['seismic_rate'][0]

        date_index_in_all = np.where((time_axis >= this_date[0]) & (time_axis <= this_date[-1]))[0]

        results.append({
            'date': this_date,
            'seismic_rate': this_seismic_rate,
            'injection_rate': this_injection_rate,
            'pressure': this_pressure,
            'injection_rate_grad': this_injection_rate_grad,
            'target': this_target,
            'prediction': this_prediction,
            'target_index': decoder_time_idx[j],
            'group_id': this_group_id,
            'date_index': date_index_in_all,
        })
    return results

def merge_and_evaluate_forecasts(results, time_axis_all, result_path=None):
    import utils
    if result_path is not None:
        os.makedirs(result_path, exist_ok=True)
        for k, rd in enumerate(results):
            save_dict = {
                'date': rd['date'], 'seismic_rate': rd['seismic_rate'],
                'pressure': rd['pressure'] if 'pressure' in rd.keys() else None,
                'injection_rate': rd['injection_rate'], 'injection_rate_grad': rd['injection_rate_grad'],
                'target': rd['target'], 'prediction': rd['prediction'],
                'target_time_index': rd['target_index'], 'group_id': rd['group_id'],
                'date_index': rd['date_index'],
            }
            np.save(os.path.join(result_path, f'forecast_window_{k:04d}.npy'), save_dict)

    tgt_list, pred_list, idx_list = [], [], []
    for rd in results:
        t_tgt = rd['date'][rd['target_index']]
        idx = np.where((time_axis_all >= t_tgt.min()) & (time_axis_all <= t_tgt.max()))[0]
        tgt_list.append(rd['target'])
        pred_list.append(rd['prediction'])
        idx_list.append(idx)
    seismic_rate_pred, global_index = utils.merge_quantile_segments(pred_list, idx_list)
    seismic_rate_tgt, _ = utils.merge_quantile_segments(tgt_list, idx_list)

    j0 = max(0, global_index[0] - len(global_index))
    global_index_inp = np.arange(j0, global_index[0], 1)

    return {
        'seismic_rate_pred': seismic_rate_pred, 'seismic_rate_tgt': seismic_rate_tgt,
        'global_index': global_index, 'global_index_inp': global_index_inp,
        'time_axis_tgt': time_axis_all[global_index],
        'time_axis_inp': time_axis_all[global_index_inp],
    }


def compute_all_window_metrics(results_sets, window_metric_names):
    if not isinstance(results_sets, dict) or len(results_sets) == 0:
        raise ValueError("results_sets must be a nonempty dictionary.")

    if len(window_metric_names) == 0:
        raise ValueError("window_metric_names must not be empty.")

    def _compute_single_window_metrics(res):
        start_time = []
        metrics = {
            name: []
            for name in window_metric_names
        }

        for i, rd in enumerate(res):
            metric_result = evaluate_time_series_similarity(
                rd["target"],
                rd["prediction"],
                verbose=False,
            )

            for name in window_metric_names:
                if name not in metric_result:
                    raise KeyError(
                        f"Evaluation results for window {i} do not contain metric: {name}"
                    )

                metrics[name].append(
                    metric_result[name]
                )

            target_index = np.asarray(
                rd["target_index"]
            ).ravel()

            if target_index.size == 0:
                raise ValueError(
                    f"Window {i} has an empty target_index."
                )

            date_array = np.asarray(rd["date"])
            first_target_index = int(target_index[0])

            start_time.append(
                date_array[first_target_index]
            )

        start_time = np.asarray(start_time)
        metrics = {
            name: np.asarray(values, dtype=float)
            for name, values in metrics.items()
        }

        return start_time, metrics

    all_window = {
        label: _compute_single_window_metrics(res)
        for label, res in results_sets.items()
    }

    return all_window


def prepare_baseline_inputs(results, meta, norm_stats, max_prediction_length, use_full_history=False):
    _driver_keys = ['injection_rate']
    if results[0].get('pressure') is not None:
        _driver_keys.append('pressure')
    _driver_keys.append('injection_rate_grad')

    def _denorm(k, a):
        return np.asarray(a, float) * norm_stats[k][1] + norm_stats[k][0]

    hist_seis = _denorm('seismic_rate', meta['seismic_rate'])
    hist_drivers = [_denorm(k, meta[k]) for k in _driver_keys]

    def _history_upto_window(rd):
        n_tgt = len(rd['target'])
        assert n_tgt == max_prediction_length, \
            f"Decoder length {n_tgt} != {max_prediction_length}"
        assert len(rd['date_index']) == len(rd['date']), \
            "date_index and date must have the same length"
        assert rd['target_index'][0] == len(rd['date']) - n_tgt, \
            "The decoder segment must be at the end of the window"

        t0 = rd['date_index'][-n_tgt]
        seis = hist_seis[:t0 + n_tgt]
        drivers = [d[:t0 + n_tgt] for d in hist_drivers]
        return seis, drivers

    def _window_only_series(rd):
        seis = np.asarray(rd['seismic_rate'], dtype=float)
        drivers = [np.asarray(rd[k], dtype=float) for k in _driver_keys]
        return seis, drivers

    if use_full_history:
        return _history_upto_window
    return _window_only_series


from concurrent.futures import ProcessPoolExecutor

def _etas_worker(args):
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    k, model_key, hist_rate, inj_hist, inj_future, cfg = args
    dt, window_size, step_size = cfg["dt"], cfg["window_size"], cfg["step_size"]
    max_pred_len = cfg["max_pred_len"]

    if model_key == "etas":
        from models.etas import ETASRollingModel
        m = ETASRollingModel(dt=dt, c=0.01, window_size=window_size, step_size=step_size)
        m.fit(hist_rate)
        pred = m.forecast(hist_rate, n_future=max_pred_len)
    elif model_key == "etas_global":
        from models.etas import fit_etas_rate_model, forecast_etas_rate
        p = fit_etas_rate_model(hist_rate, dt=dt)
        pred = forecast_etas_rate(hist_rate, p, n_forecast=max_pred_len, dt=dt)
    elif model_key == "mancini":
        from models.etas_mancini import InjectionDrivenETASRollingModel
        m = InjectionDrivenETASRollingModel(dt=dt, c=0.01, window_size=window_size, step_size=step_size)
        m.fit(hist_rate, inj_hist)
        pred = m.forecast(hist_rate, inj_hist, inj_future)
    elif model_key == "kim":
        from models.etas_kim_convolution import ConvolutionalInjectionModel
        m = ConvolutionalInjectionModel(dt=dt)
        m.fit(hist_rate, inj_hist)
        pred = m.forecast(inj_hist, inj_future)
    else:
        raise ValueError(model_key)
    return k, np.asarray(pred, dtype=float)

def _prep_window_inputs(rd, get_inputs_fn, i_inj, max_pred_len):
    seis, drivers = get_inputs_fn(rd)
    hist_rate = np.asarray(seis[:-max_pred_len], dtype=float)
    inj = np.asarray(drivers[i_inj], dtype=float)
    return hist_rate, inj[:-max_pred_len], inj[-max_pred_len:]

def run_etas_parallel(model_key, results, get_inputs_fn, max_pred_len, dt,
                      driver_keys, window_size=8, step_size=2,
                      n_workers=None, use_parallel=True, progress_name=None):
    i_inj = driver_keys.index("injection_rate")
    cfg = {"dt": dt, "window_size": window_size, "step_size": step_size, "max_pred_len": max_pred_len}
    window_inputs = [_prep_window_inputs(rd, get_inputs_fn, i_inj, max_pred_len) for rd in results]
    args_list = [(k, model_key, wi[0], wi[1], wi[2], cfg) for k, wi in enumerate(window_inputs)]
    n = len(args_list)

    preds = [None] * n
    iterator = (_etas_worker(a) for a in args_list)
    ex = None
    if use_parallel and n_workers != 1:
        nw = n_workers or min(os.cpu_count() or 4, 64)
        ex = ProcessPoolExecutor(max_workers=nw)
        iterator = ex.map(_etas_worker, args_list)
    try:
        for k, pred in iterator:
            preds[k] = pred
            if progress_name and (k + 1) % 50 == 0:
                print(f"{progress_name}: {k+1}/{n} windows completed", flush=True)
    finally:
        if ex is not None:
            ex.shutdown(wait=True)

    out = []
    for rd, pred in zip(results, preds):
        new = dict(rd)
        new["prediction"] = pred
        out.append(new)
    return out


def _fit_etas_once(model_key, fit_rate, fit_inj, dt, window_size, step_size):
    if model_key == "etas":
        from models.etas import ETASRollingModel
        m = ETASRollingModel(dt=dt, c=0.01, window_size=window_size, step_size=step_size)
        m.fit(fit_rate)
        return m.param_series
    elif model_key == "etas_global":
        from models.etas import fit_etas_rate_model
        return fit_etas_rate_model(fit_rate, dt=dt)
    elif model_key == "mancini":
        from models.etas_mancini import InjectionDrivenETASRollingModel
        m = InjectionDrivenETASRollingModel(dt=dt, c=0.01, window_size=window_size, step_size=step_size)
        m.fit(fit_rate, fit_inj)
        return m.param_series
    elif model_key == "kim":
        from models.etas_kim_convolution import ConvolutionalInjectionModel
        m = ConvolutionalInjectionModel(dt=dt)
        m.fit(fit_rate, fit_inj)
        return m.params
    else:
        raise ValueError(model_key)


def _etas_frozen_worker(args):
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    k, model_key, hist_rate, inj_hist, inj_future, params, cfg = args
    dt, window_size, step_size = cfg["dt"], cfg["window_size"], cfg["step_size"]
    max_pred_len = cfg["max_pred_len"]

    if model_key == "etas":
        from models.etas import ETASRollingModel
        m = ETASRollingModel(dt=dt, c=0.01, window_size=window_size, step_size=step_size)
        m.param_series = params
        pred = m.forecast(hist_rate, n_future=max_pred_len, refit=False)
    elif model_key == "etas_global":
        from models.etas import forecast_etas_rate
        pred = forecast_etas_rate(hist_rate, params, n_forecast=max_pred_len, dt=dt)
    elif model_key == "mancini":
        from models.etas_mancini import InjectionDrivenETASRollingModel
        m = InjectionDrivenETASRollingModel(dt=dt, c=0.01, window_size=window_size, step_size=step_size)
        m.param_series = params
        pred = m.forecast(hist_rate, inj_hist, inj_future, n_future=max_pred_len, refit=False)
    elif model_key == "kim":
        from models.etas_kim_convolution import ConvolutionalInjectionModel
        m = ConvolutionalInjectionModel(dt=dt)
        m.params = params
        pred = m.forecast(inj_hist, inj_future)
    else:
        raise ValueError(model_key)
    return k, np.asarray(pred, dtype=float)


def run_etas_frozen(model_key, results, get_inputs_fn, fit_rate, fit_inj, max_pred_len, dt,
                    driver_keys, window_size=8, step_size=2,
                    n_workers=None, use_parallel=True, progress_name=None):
    i_inj = driver_keys.index("injection_rate")
    cfg = {"dt": dt, "window_size": window_size, "step_size": step_size, "max_pred_len": max_pred_len}

    t0 = time.time()
    params = _fit_etas_once(model_key, fit_rate, fit_inj, dt, window_size, step_size)
    if progress_name:
        print(f"{progress_name}: pre-zero fit completed ({time.time()-t0:.1f}s); starting forecasts with fixed parameters", flush=True)

    window_inputs = [_prep_window_inputs(rd, get_inputs_fn, i_inj, max_pred_len) for rd in results]
    args_list = [(k, model_key, wi[0], wi[1], wi[2], params, cfg) for k, wi in enumerate(window_inputs)]
    n = len(args_list)

    preds = [None] * n
    iterator = (_etas_frozen_worker(a) for a in args_list)
    ex = None
    if use_parallel and n_workers != 1:
        nw = n_workers or min(os.cpu_count() or 4, 64)
        ex = ProcessPoolExecutor(max_workers=nw)
        iterator = ex.map(_etas_frozen_worker, args_list)
    try:
        for k, pred in iterator:
            preds[k] = pred
            if progress_name and (k + 1) % 50 == 0:
                print(f"{progress_name}: {k+1}/{n} windows completed", flush=True)
    finally:
        if ex is not None:
            ex.shutdown(wait=True)

    out = []
    for rd, pred in zip(results, preds):
        new = dict(rd)
        new["prediction"] = pred
        out.append(new)
    return out

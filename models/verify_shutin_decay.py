import argparse
import os
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import numpy as np
import torch
from datetime import datetime, timedelta

import utils
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder

from models.verify_shutin_bias import build_data_lists
from models.shutin_decay_tft import ShutInDecayEstimator, ShutInDecayTFT

THRESHOLD_BBL = 1.0
STEP_HOURS = 480.0 / 3600.0
LOG_NAME = 'utahforge2024_i200o80t2024-04-06-06-06_dc'
ZERO_DATE = datetime(2024, 4, 6, 6, 6)


def load_raw_series(d, downsample_rate=4, with_norm=False):
    seis, inj, pres = d['seis_rate'], d['inj_rate'], d['pres']
    ta = d['time_axis']
    out = utils.normalize_features(seis, inj, np.gradient(inj), pressure=pres,
                                   pressure_grad=np.gradient(pres), normm='min_max')
    seis_n, inj_n, ig_n, p_n, pg_n, norm_param = out
    step_s = downsample_rate * int((ta[1] - ta[0]).total_seconds())
    ta_d, seis_d, inj_d, *_ = utils.downsample_time_series(
        ta, seis_n, inj_n, ig_n, p_n, pg_n, rule=f"{step_s}S")
    seis_range = float(norm_param['seismic_rate']['range'])
    inj_range = float(norm_param['injection_rate']['range'])
    ret = (np.asarray(ta_d), np.asarray(seis_d, float) * seis_range,
           np.asarray(inj_d, float) * inj_range, seis_range)
    return ret + (norm_param,) if with_norm else ret


def build_notebook_windows():
    d2024 = np.load('preparation/dataset/UtahForge2024/seismic_inject_dataset2024_long.npy',
                    allow_pickle=True).item()
    tr24, va24, ap24, _ = build_data_lists(d2024, 'min_max', 4, 200, 80, ZERO_DATE)
    gid = 0
    for seg in tr24 + va24 + ap24:
        seg['group_id'] = gid
        gid += 1
        enc = seg['seismic_rate'][:200]
        seg['enc_seis_mean'] = float(enc.mean())
        seg['enc_seis_tail20'] = float(enc[-20:].mean())
    return tr24, va24, ap24


def make_datasets(train_list, train_df_target, known, ckpt):
    train_df = utils.lists_to_df(train_list, train_df_target, known)
    training = TimeSeriesDataSet(
        train_df, time_idx="time_idx", target=train_df_target, group_ids=["group_id"],
        min_encoder_length=100, max_encoder_length=200,
        min_prediction_length=80, max_prediction_length=80,
        static_categoricals=[], static_reals=["enc_seis_mean", "enc_seis_tail20"],
        scalers={"enc_seis_mean": None, "enc_seis_tail20": None},
        time_varying_known_categoricals=[], time_varying_known_reals=known,
        time_varying_unknown_categoricals=[], time_varying_unknown_reals=[train_df_target],
        categorical_encoders={"group_id": NaNLabelEncoder(add_nan=True)},
        target_normalizer=None, add_relative_time_idx=True,
        add_target_scales=True, add_encoder_length=True)
    hp = torch.load(ckpt, map_location='cpu', weights_only=False)['hyper_parameters']
    for k, sc in hp['dataset_parameters']['scalers'].items():
        training.scalers[k] = sc
    return training


def stratified_bias_table(rows, title):
    print(f"\n===== {title} =====")
    print(f"{'Interval':<12}{'n':>8}{'Observed':>9}{'Predicted':>9}{'Bias':>10}")
    for lab, n, tg, pr, bias in rows:
        print(f"{lab:<12}{n:>8d}{tg:>9.4f}{pr:>9.4f}{bias:>+10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--stride', type=int, default=1, help='Subsampling stride for application windows (for debugging)')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--save-cache', action='store_true',
                    help='With stride=1, follow the notebook workflow to generate '
                         'tft_shutin_decay_results.npy and compare combined scores')
    args = ap.parse_args()

    if args.ckpt is None:
        log_path = f'./logs/{LOG_NAME}'
        versions = sorted([v for v in os.listdir(log_path) if v.startswith('version')],
                          key=lambda x: int(x.split('_')[-1]))
        ckpt_dir = os.path.join(log_path, versions[-1], 'ckpts')
        ckpts = sorted([x for x in os.listdir(ckpt_dir) if x.endswith('.ckpt')],
                       key=lambda x: int(x.split('=')[1].split('-')[0]))
        args.ckpt = os.path.join(ckpt_dir, ckpts[-1])
    print(f"checkpoint: {args.ckpt}")

    tr24, va24, ap24 = build_notebook_windows()
    ta_d, seis_raw, inj_raw, seis_range = load_raw_series(
        np.load('preparation/dataset/UtahForge2024/seismic_inject_dataset2024_long.npy',
                allow_pickle=True).item())
    zero_index = int(np.where(ta_d >= ZERO_DATE)[0][0])
    print(f"zero_index={zero_index}, apply windows={len(ap24)}, seis_range={seis_range:.4f}")

    target = "seismic_rate"
    known = ["injection_rate", "injection_rate_grad", "pressure"]
    training = make_datasets(tr24, target, known, args.ckpt)

    model = ShutInDecayTFT.load_from_checkpoint(args.ckpt, map_location='cpu',
                                                weights_only=False)
    model.eval()

    _d24raw = np.load('preparation/dataset/UtahForge2024/seismic_inject_dataset2024_long.npy',
                      allow_pickle=True).item()
    _ta_raw = np.asarray(_d24raw['time_axis'])
    _seis_2min = np.asarray(_d24raw['seis_rate'], float)[:int(np.searchsorted(
        np.array([np.datetime64(t) for t in _ta_raw]), np.datetime64(ZERO_DATE)))]
    _pos = np.unique(_seis_2min[_seis_2min > 0])
    if len(_pos) >= 2:
        _dif = np.diff(_pos)
        quantum = float(_dif[_dif > 1e-9].min())
    elif len(_pos):
        quantum = float(_pos[0])
    else:
        quantum = None
    print(f"detection_quantum (2-min raw pre-zero, grid spacing) = {quantum}")
    est = ShutInDecayEstimator(threshold_bbl=THRESHOLD_BBL, step_hours=STEP_HOURS,
                               detection_quantum=quantum)
    l2022 = np.load('preparation/dataset/UtahForge2022/seismic_inject_dataset2022_ext.npy',
                    allow_pickle=True).tolist()
    for seg in l2022:
        _, seis22, inj22, _ = load_raw_series(seg)
        est.add_pool_segment(inj22, seis22)
    est.set_application_series(inj_raw, seis_raw, zero_index)

    origins_all = zero_index + np.arange(len(ap24))
    sel = np.arange(0, len(ap24), args.stride)
    sub_list = [dict(ap24[i]) for i in sel]
    sub_df = utils.lists_to_df(sub_list, target, known)
    application = TimeSeriesDataSet.from_dataset(training, sub_df, predict=True,
                                                 stop_randomization=True)
    loader = application.to_dataloader(train=False, batch_size=args.batch_size,
                                       num_workers=0)
    model.prime(est, origins_all[sel], seis_range)
    raw = model.predict(loader, mode="raw", return_x=True)
    pred = raw.output["prediction"].detach().cpu().numpy() * seis_range
    tgt = raw.x["decoder_target"].detach().cpu().numpy() * seis_range
    pred_mean = pred.mean(axis=-1)

    cache = os.path.join('results', 'utahforge2024', LOG_NAME, 'tft_current_results.npy')
    cur = np.load(cache, allow_pickle=True)
    cur_tgt = np.stack([cur[j]['target'] for j in sel])
    cur_pred = np.stack([cur[j]['prediction'] for j in sel]).mean(axis=-1)
    assert np.allclose(cur_tgt, tgt, atol=1e-4), "Cached targets do not match the new inference targets"

    D = np.zeros(len(inj_raw))
    cnt = 0.0
    for i in range(len(inj_raw)):
        cnt = cnt + 1 if inj_raw[i] <= THRESHOLD_BBL else 0.0
        D[i] = cnt
    dt_dec = np.stack([D[o:o + 80] for o in origins_all[sel]])
    gate = dt_dec > 0

    def rows_for(pm):
        rows = []
        m = ~gate
        rows.append(('Injection on', int(m.sum()), tgt[m].mean(), pm[m].mean(), (pm - tgt)[m].mean()))
        for lo, hi, lab in [(1, 2, 'Onset(dt<=2)'), (3, 12, '0.3-1.6h'),
                            (13, 48, '1.6-6.4h'), (49, 96, '6.4-12.8h'),
                            (97, 1 << 30, '>12.8h')]:
            m = gate & (dt_dec >= lo) & (dt_dec <= hi)
            if m.sum():
                rows.append((lab, int(m.sum()), tgt[m].mean(), pm[m].mean(),
                             (pm - tgt)[m].mean()))
        rows.append(('Overall MAE', int(tgt.size), tgt.mean(), pm.mean(),
                     np.abs(pm - tgt).mean()))
        return rows

    stratified_bias_table(rows_for(cur_pred), 'current TFT (latest _dc checkpoint)')
    stratified_bias_table(rows_for(pred_mean), 'TFT + shutin-decay (this module)')

    bias_c = pred_mean - tgt
    bias_cur = cur_pred - tgt
    m_late = gate & (dt_dec >= 97)
    gate_off_maxdiff = float(np.abs(pred_mean[~gate] - cur_pred[~gate]).max())
    t_idx = np.arange(80)[None, :]
    m_onset = gate & (dt_dec < t_idx + 2)
    onset_maxdiff = float(np.abs(pred_mean[m_onset] - cur_pred[m_onset]).max()) if m_onset.any() else 0.0
    print("\n===== Validation (ADR-0004, preorigin-only) =====")
    checks = [
        (">12.8h |bias| ≤ 0.03", abs(bias_c[m_late].mean()) <= 0.03,
         f"{bias_c[m_late].mean():+.4f} (current {bias_cur[m_late].mean():+.4f})"),
        ("Injection-on predictions match current elementwise within float32 precision", gate_off_maxdiff < 1e-5,
         f"max diff = {gate_off_maxdiff:.2e}"),
        ("Mid-decoder onset predictions match current elementwise within float32 precision", onset_maxdiff < 1e-5,
         f"max diff = {onset_maxdiff:.2e}"),
        ("Overall MAE improved", np.abs(bias_c).mean() < np.abs(bias_cur).mean(),
         f"{np.abs(bias_c).mean():.4f} vs {np.abs(bias_cur).mean():.4f}"),
    ]
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")

    wcs = [wc for _, wc in model.primed_windows]
    w_arr = np.array([wc.w if wc.a_ep is not None else np.nan for wc in wcs])
    have = np.isfinite(w_arr)
    n_early = max(1, int(round(0.25 * len(sel))))
    early_m = np.zeros(len(sel), bool)
    early_m[:n_early] = True
    print("\n===== v2 diagnostics (ADR-0007) =====")
    if have.any():
        q = np.nanpercentile(w_arr, [10, 50, 90])
        print(f"A2 weight w (n={have.sum()} ongoing windows): p10/p50/p90 = "
              f"{q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}; "
              f"early mean {np.nanmean(w_arr[early_m]):.3f}, "
              f"late mean {np.nanmean(w_arr[~early_m]):.3f}")

    floored = np.zeros_like(gate)
    curve_val = np.zeros_like(tgt)
    blend_val = np.zeros_like(tgt)
    for wi, (origin, wc) in enumerate(model.primed_windows):
        if wc.a_ep is None:
            continue
        for t in range(80):
            i = origin + t
            if i >= len(inj_raw) or inj_raw[i] > THRESHOLD_BBL:
                continue
            s0 = est.run_start_of(i)
            if s0 < 0 or s0 >= origin:
                continue
            dth = (i - s0 + 1) * STEP_HOURS
            _g = float(wc.g(np.array([dth]))[0])
            if dth > wc.dt_val_hours:
                _ge = float(wc.g(np.array([wc.dt_val_hours]))[0])
                _ga = wc.dt_val_hours / dth
                _g = _ga * _g + (1.0 - _ga) * _ge
            curve = wc.a_ep * _g
            curve_val[wi, t] = curve
            blend_val[wi, t] = wc.w * curve + (1.0 - wc.w) * cur_pred[wi, t]
            if wc.quiet and wc.floor_raw > 0 and curve < wc.floor_raw:
                floored[wi, t] = True

    def _smape(a, b):
        return 200.0 * np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12)

    b1_net_ok = True
    b1_msg = "B1 not triggered (0 steps)"
    if floored.any():
        obs = tgt[floored]
        sm_floor = _smape(obs, np.zeros_like(obs))
        sm_unfl = _smape(obs, blend_val[floored])
        n0 = int((obs == 0).sum())
        b1_net_ok = sm_floor.mean() < sm_unfl.mean()
        b1_msg = (f"Steps set to zero: {int(floored.sum())} (obs==0: {n0}, "
                  f"positive observations affected (obs>0): {int(floored.sum()) - n0}); "
                  f"mean per-step SMAPE with zeroing {sm_floor.mean():.1f} vs without zeroing {sm_unfl.mean():.1f}")
        fl_early = int(floored[early_m].sum())
        print(f"B1: {b1_msg}; triggered steps in early windows = {fl_early} (expected 0)")
        b1_net_ok = b1_net_ok and fl_early == 0
    print(f"  [{'PASS' if b1_net_ok else 'FAIL'}] B1 provides a net benefit with no triggers in early windows: {b1_msg}")

    print("\ng(Δt) parameters across sampled origins:")
    for w in np.linspace(0, len(sel) - 1, 5).astype(int):
        wc = model.primed_windows[w][1]
        o = int(origins_all[sel][w])
        print(f"  window {w:4d} origin={o} ({ta_d[o]}): kind={wc.g.kind} "
              f"params={np.round(wc.g.params_vector, 3).tolist()} scale={wc.g.scale:.3f} "
              f"A_ep={None if wc.a_ep is None else round(wc.a_ep, 4)}")

    if args.save_cache:
        assert args.stride == 1, "--save-cache requires stride=1 (matching the notebook workflow)"
        _, _, _, _, norm_param = load_raw_series(
            np.load('preparation/dataset/UtahForge2024/seismic_inject_dataset2024_long.npy',
                    allow_pickle=True).item(), with_norm=True)
        norm_stats = {k: np.asarray(list(v.values())) for k, v in norm_param.items()}
        results_decay = utils.predict_and_collect_results(
            model, loader, sub_df, application, ta_d, norm_stats)
        cache_out = os.path.join('results', 'utahforge2024', LOG_NAME,
                                 'tft_shutin_decay_results.npy')
        np.save(cache_out, results_decay)
        print(f"\n[saved] results_decay ({len(results_decay)} windows) -> {cache_out}")

        rdir = os.path.join('results', 'utahforge2024', LOG_NAME)
        results_sets = {
            'current': np.load(os.path.join(rdir, 'tft_current_results.npy'), allow_pickle=True),
            'arx': np.load(os.path.join(rdir, 'arx_pz_results.npy'), allow_pickle=True),
            'gbdt': np.load(os.path.join(rdir, 'gbdt_pz_results.npy'), allow_pickle=True),
            'etas': np.load(os.path.join(rdir, 'etas_pz_results.npy'), allow_pickle=True),
            'etas_global': np.load(os.path.join(rdir, 'etas_global_pz_results.npy'), allow_pickle=True),
            'etas_mancini': np.load(os.path.join(rdir, 'etas_mancini_pz_results.npy'), allow_pickle=True),
            'etas_kim': np.load(os.path.join(rdir, 'etas_kim_pz_results.npy'), allow_pickle=True),
            'tft_shutin_decay': results_decay,
        }
        metric_names = ["RMSE", "MAE", "Pearson_correlation", "Cosine_similarity", "SMAPE_%"]
        all_window = utils.compute_all_window_metrics(
            results_sets=results_sets, window_metric_names=metric_names)
        models = list(results_sets.keys())
        start = np.asarray(all_window['current'][0])
        order = np.argsort(start)
        n_early = max(1, int(round(0.25 * len(start))))
        early_idx = order[:n_early]
        G = []
        for mn in metric_names:
            M = np.column_stack([all_window[m][1][mn] for m in models])
            vmin = np.nanmin(M, axis=1, keepdims=True)
            vmax = np.nanmax(M, axis=1, keepdims=True)
            rng = np.where(vmax - vmin > 0, vmax - vmin, 1.0)
            g = (vmax - M) / rng if mn in {"RMSE", "MAE", "SMAPE_%"} else (M - vmin) / rng
            g[~np.isfinite(M)] = np.nan
            G.append(g)
        cs = np.nanmean(np.stack(G, axis=0), axis=0)
        print(f"\ncombined score (early = first {n_early}/{len(start)} windows):")
        print(f"{'model':<18}{'overall':>10}{'early':>10}")
        for i, m in enumerate(models):
            print(f"{m:<18}{np.nanmean(cs[:, i]):>10.4f}{np.nanmean(cs[early_idx, i]):>10.4f}")
        i_dec = models.index('tft_shutin_decay')
        ov = np.nanmean(cs, axis=0)
        ea = np.nanmean(cs[early_idx], axis=0)
        ok_ov = np.argmax(ov) == i_dec
        ok_ea = np.argmax(ea) == i_dec
        print(f"  [{'PASS' if ok_ov else 'FAIL'}] overall ranked first (ADR-0007 Validation 1): "
              f"decay {ov[i_dec]:.4f}, best competing model "
              f"{models[int(np.argsort(ov)[-2 if ok_ov else -1])]} "
              f"{np.sort(ov)[-2 if ok_ov else -1]:.4f}")
        print(f"  [{'PASS' if ok_ea else 'FAIL'}] early ranked first (ADR-0007 Validation 1): "
              f"decay {ea[i_dec]:.4f}, best competing model "
              f"{models[int(np.argsort(ea)[-2 if ok_ea else -1])]} "
              f"{np.sort(ea)[-2 if ok_ea else -1]:.4f}")


if __name__ == '__main__':
    main()

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from datetime import datetime, timedelta

import utils
from pytorch_forecasting import TimeSeriesDataSet


def get_utahforge_segments(border1s, border2s, time_axis, seismic_rate, injection_rate,
    injection_rate_grad, pressure, pressure_grad, w_len, n_len, max_encoder_length,
    max_prediction_length, w_step=1):
    border1, border2 = border1s[0], border2s[0]
    train_data_list, icount = [], 0
    for j0 in range(border1, border2 + 1, w_step):
        if j0 < 0:
            continue
        j1 = j0 + w_len
        if j1 > n_len:
            j0 = n_len - w_len
            j1 = n_len
            if j0 < 0:
                break
        train_data_list.append({
            'seismic_rate': seismic_rate[j0:j1], 'pressure': pressure[j0:j1],
            'injection_rate': injection_rate[j0:j1], 'pressure_grad': pressure_grad[j0:j1],
            'injection_rate_grad': injection_rate_grad[j0:j1],
            'date': np.array([x for x in time_axis[j0:j1]]),
            'time_idx': np.arange(j1 - j0, dtype=np.int64), 'group_id': icount})
        icount += 1
        if j1 == n_len:
            break
    border1, border2 = border1s[1], border2s[1]
    valid_data_list = []
    w_len = max_encoder_length + max_prediction_length
    for j0 in range(border1, border2 + 1, w_step):
        if j0 < 0:
            continue
        j1 = j0 + w_len
        if j1 > n_len:
            j0 = n_len - w_len
            j1 = n_len
            if j0 < 0:
                break
        valid_data_list.append({
            'seismic_rate': seismic_rate[j0:j1], 'pressure': pressure[j0:j1],
            'injection_rate': injection_rate[j0:j1], 'pressure_grad': pressure_grad[j0:j1],
            'injection_rate_grad': injection_rate_grad[j0:j1],
            'date': np.array([x for x in time_axis[j0:j1]]),
            'time_idx': np.arange(j1 - j0, dtype=np.int64), 'group_id': icount})
        icount += 1
        if j1 == n_len:
            break
    return train_data_list, valid_data_list


def build_data_lists(data_dict, normm, downsample_rate, max_encoder_length,
                     max_prediction_length, zero_date, w_step=1):
    seismic_rate = data_dict['seis_rate']
    injection_rate = data_dict['inj_rate']
    pressure = data_dict['pres']
    injection_rate_grad = np.gradient(injection_rate)
    pressure_grad = np.gradient(pressure)
    time_axis = data_dict['time_axis']
    interval_seconds_original = int((time_axis[1] - time_axis[0]).total_seconds())
    (seismic_rate, injection_rate, injection_rate_grad, pressure,
     pressure_grad, norm_param) = utils.normalize_features(
        seismic_rate, injection_rate, injection_rate_grad,
        pressure=pressure, pressure_grad=pressure_grad, normm=normm)
    interval_seconds = downsample_rate * interval_seconds_original
    (time_axis, seismic_rate, injection_rate, injection_rate_grad,
     pressure, pressure_grad) = utils.downsample_time_series(
        time_axis, seismic_rate, injection_rate, injection_rate_grad,
        pressure, pressure_grad, rule=f"{interval_seconds}S")
    n_len = len(time_axis)
    w_len = max_encoder_length + max_prediction_length
    if zero_date is None:
        zero_index = n_len - 1
    else:
        zero_index = np.where((time_axis >= zero_date)
                              & (time_axis < zero_date + timedelta(seconds=interval_seconds)))[0][0]
    border1s = [0, zero_index - w_len]
    border2s = [zero_index - w_len, n_len - 1]
    train_data_list, valid_data_list = get_utahforge_segments(
        border1s, border2s, time_axis, seismic_rate, injection_rate, injection_rate_grad,
        pressure, pressure_grad, w_len, n_len, max_encoder_length, max_prediction_length,
        w_step=w_step)
    apply_data_list = valid_data_list[max_prediction_length:]
    valid_data_list = valid_data_list[:max_prediction_length]
    meta = {'time_axis': time_axis, 'norm_param': norm_param}
    return train_data_list, valid_data_list, apply_data_list, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-name', default='utahforge2024_i200o80t2024-04-06-06-06_0718')
    ap.add_argument('--normm', default='min_max', choices=['mea_std', 'min_max'])
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--ckpt', default=None, help='默认取 log 目录最新 version 的 ckpts 里最后的 checkpoint')
    args = ap.parse_args()

    max_encoder_length, max_prediction_length, downsample_rate = 200, 80, 4
    zero_date = datetime(2024, 4, 6, 6, 6)

    import os
    if args.ckpt is None:
        log_path = f'./logs/{args.log_name}'
        versions = sorted([v for v in os.listdir(log_path) if v.startswith('version')],
                          key=lambda x: int(x.split('_')[-1]))
        ckpt_dir = os.path.join(log_path, versions[-1], 'ckpts')
        ckpts = sorted([x for x in os.listdir(ckpt_dir) if x.endswith('.ckpt')],
                       key=lambda x: int(x.split('=')[1].split('-')[0]))
        args.ckpt = os.path.join(ckpt_dir, ckpts[-1])
    print(f"checkpoint: {args.ckpt}")

    d2024 = np.load('preparation/dataset/UtahForge2024/seismic_inject_dataset2024_ext.npy',
                    allow_pickle=True).item()
    tr24, va24, ap24, meta = build_data_lists(
        d2024, args.normm, downsample_rate, max_encoder_length, max_prediction_length, zero_date)
    l2022 = np.load('preparation/dataset/UtahForge2022/seismic_inject_dataset2022_ext.npy',
                    allow_pickle=True).tolist()
    tr22 = []
    for seg in l2022:
        t, v, _, _ = build_data_lists(seg, args.normm, downsample_rate,
                                      max_encoder_length, max_prediction_length, None)
        tr22 += t + v
    train_list = tr22 + tr24
    for n, s in enumerate(train_list + va24 + ap24):
        s['group_id'] = n

    target = "seismic_rate"
    known = ["injection_rate", "injection_rate_grad", "pressure"]
    train_df = utils.lists_to_df(train_list, target, known)
    training = TimeSeriesDataSet(
        train_df, time_idx="time_idx", target=target, group_ids=["group_id"],
        min_encoder_length=100, max_encoder_length=200,
        min_prediction_length=80, max_prediction_length=80,
        static_categoricals=[], static_reals=[],
        time_varying_known_categoricals=[], time_varying_known_reals=known,
        time_varying_unknown_categoricals=[], time_varying_unknown_reals=[target],
        target_normalizer=None, add_relative_time_idx=True,
        add_target_scales=True, add_encoder_length=True)

    hp = torch.load(args.ckpt, map_location='cpu', weights_only=False)['hyper_parameters']
    for k, sc in hp['dataset_parameters']['scalers'].items():
        training.scalers[k] = sc

    s_seis = np.asarray(list(meta['norm_param']['seismic_rate'].values()))
    s_inj = np.asarray(list(meta['norm_param']['injection_rate'].values()))
    inj_sc = training.scalers['injection_rate']
    gate_z1 = 0.01 * float(inj_sc.scale_) + float(inj_sc.mean_)
    print(f"normm={args.normm} | 门限 z2<0.01 <=> raw injection < {gate_z1 * s_inj[1] + s_inj[0]:.2f} bbl/min")

    sub_list = []
    for n, i in enumerate(range(0, len(ap24), args.stride)):
        s = dict(ap24[i])
        s['group_id'] = n
        sub_list.append(s)
    sub_df = utils.lists_to_df(sub_list, target, known)
    application = TimeSeriesDataSet.from_dataset(training, sub_df, predict=True,
                                                 stop_randomization=True)
    loader = application.to_dataloader(train=False, batch_size=64, num_workers=0)

    from models.physics_tft_model import PhysicsAugmentedTFT
    model = PhysicsAugmentedTFT.load_from_checkpoint(args.ckpt, map_location='cpu',
                                                     weights_only=False)
    model.eval()
    omori_out = []
    hook = model.omori_net.register_forward_hook(lambda m, i, o: omori_out.append(o.detach().cpu()))
    preds, tgts, decs, encs = [], [], [], []
    with torch.no_grad():
        for x, _ in loader:
            out = model(x)
            preds.append(out['prediction'].cpu().numpy())
            tgts.append(x['decoder_target'].cpu().numpy())
            decs.append(x['decoder_cont'].cpu().numpy())
            encs.append(x['encoder_cont'].cpu().numpy())
    hook.remove()

    pred = np.concatenate(preds)
    tgt = np.concatenate(tgts)
    dec_cont = np.concatenate(decs)
    enc_cont = np.concatenate(encs)
    oo = torch.cat(omori_out).numpy()
    K = 1 / (1 + np.exp(-oo[..., 0]))
    inj_dec = dec_cont[..., 3]
    inj_all = np.concatenate([enc_cont[..., 3], inj_dec], axis=1)
    gate = inj_dec < 0.01
    dt = np.zeros_like(inj_all)
    cnt = np.zeros(inj_all.shape[0])
    for t in range(inj_all.shape[1]):
        cnt = np.where(inj_all[:, t] < 0.01, cnt + 1, 0.0)
        dt[:, t] = cnt
    dt_dec = dt[:, 200:]

    pred_mean = pred.mean(axis=-1)
    bias = (pred_mean - tgt) * s_seis[1]
    tgt_raw = tgt * s_seis[1] + s_seis[0]

    print(f"\noverall: bias={bias.mean():+.4f} | MAE={np.abs(bias).mean():.4f} | tgt mean={tgt_raw.mean():.4f}")
    print(f"gate ON : {gate.mean() * 100:.1f}% steps | bias={bias[gate].mean():+.4f} "
          f"| tgt={tgt_raw[gate].mean():.4f} pred={(pred_mean[gate] * s_seis[1] + s_seis[0]).mean():.4f}")
    print(f"gate OFF: bias={bias[~gate].mean():+.4f} | tgt={tgt_raw[~gate].mean():.4f}")
    print("gate-ON by shut-in duration:")
    k_late = None
    for lo, hi, lab in [(1, 12, '0-1.6h'), (13, 48, '1.6-6.4h'),
                        (49, 96, '6.4-12.8h'), (97, 280, '>12.8h')]:
        m = gate & (dt_dec >= lo) & (dt_dec <= hi)
        if m.sum() == 0:
            continue
        if lab == '>12.8h':
            k_late = K[m].mean()
            bias_late = bias[m].mean()
        print(f"  dt {lab:>9}: n={m.sum():6d} bias={bias[m].mean():+.4f} "
              f"tgt={tgt_raw[m].mean():.4f} pred={(pred_mean[m] * s_seis[1] + s_seis[0]).mean():.4f} "
              f"K={K[m].mean():.3f}")

    print("\n===== 验收 (docs/proposals/physics-tft-shutin-overprediction.md §5) =====")
    checks = [
        (">12.8h 偏差 < +0.03", bias_late < 0.03, f"{bias_late:+.4f}"),
        (">12.8h K 均值 > 0.2 (模块存活)", k_late > 0.2, f"{k_late:.3f}"),
        ("gate OFF 偏差 |.| < 0.03 (无回归)", abs(bias[~gate].mean()) < 0.03, f"{bias[~gate].mean():+.4f}"),
    ]
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {val}")


if __name__ == '__main__':
    main()

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer as BaseTemporalFusionTransformer,
)

class CompositeQuantileLoss(QuantileLoss):
    def __init__(self, quantiles=None, alpha: float = 0.0, **kwargs):
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(quantiles=quantiles, **kwargs)
        self.alpha = float(alpha)

    def loss(self, y_pred, target):
        losses = super().loss(y_pred, target)
        if self.alpha > 0:
            point = y_pred.mean(dim=-1)
            mse = (point - target) ** 2 / y_pred.size(-1)
            losses = losses + self.alpha * mse.unsqueeze(-1)
        return losses


class TemporalFusionTransformerNew(BaseTemporalFusionTransformer):

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        return super().from_dataset(dataset, **kwargs)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, weights_only=False, **kwargs):
        return super().load_from_checkpoint(
            checkpoint_path, weights_only=weights_only, **kwargs)

    def configure_optimizers(self):
        out = super().configure_optimizers()
        if isinstance(out, dict) and not out.get("lr_scheduler"):
            return out["optimizer"]
        return out

    def predict(self, *args, trainer_kwargs=None, **kwargs):
        tk = {"devices": 1, "logger": False}
        if trainer_kwargs:
            tk.update(trainer_kwargs)
        return super().predict(*args, trainer_kwargs=tk, **kwargs)

    def finetune_on(self, dataloader, lr: float, max_steps: int,
                    gradient_clip_val: float = 0.1, accelerator: str = "auto"):
        import lightning.pytorch as pl

        saved = {k: self.hparams.get(k) for k in
                 ("learning_rate", "optimizer", "reduce_on_plateau_patience")}
        self.hparams["learning_rate"] = lr
        self.hparams["optimizer"] = "adam"
        self.hparams["reduce_on_plateau_patience"] = None
        saved["log_interval"] = self.hparams.get("log_interval")
        self.hparams["log_interval"] = -1
        try:
            trainer = pl.Trainer(
                max_steps=max_steps,
                max_epochs=-1,
                accelerator=accelerator,
                devices=1,
                gradient_clip_val=gradient_clip_val,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                num_sanity_val_steps=0,
            )
            trainer.fit(self, train_dataloaders=dataloader)
        finally:
            for k, v in saved.items():
                self.hparams[k] = v
        return self


class TemporalFusionTransformerDelta(TemporalFusionTransformerNew):

    def forward(self, x):
        out = super().forward(x)
        enc_t = x["encoder_target"]
        if isinstance(enc_t, (list, tuple)):
            enc_t = enc_t[0]
        idx = (x["encoder_lengths"] - 1).clamp(min=0)
        anchor = enc_t.gather(1, idx.unsqueeze(-1)).squeeze(-1)
        return out._replace(
            prediction=out["prediction"] + anchor[:, None, None])


@dataclass
class RollingConfig:

    refit_every: int = 100000000
    ft_lr: float = 3e-3
    ft_steps: int = 100
    alpha: float = 0.0
    prefix_finetune: bool = False
    l_min_prefix: int = 4
    batch_size: int = 1024
    num_workers: int = 0
    accelerator: str = "auto"
    seed: int = 20260806
    verbose: bool = True


class RollingEnsembleForecaster:

    def __init__(self, config: RollingConfig | None = None):
        self.config = config if config is not None else RollingConfig()
        self.info_: dict = {}


    @staticmethod
    def _gate(pool, n_train) -> bool:
        return bool(pool)


    @staticmethod
    def origin_step_indices(data_list, horizon: int):
        dates = [pd.to_datetime(np.asarray(seg["date"])) for seg in data_list]
        step = pd.Timedelta(int(np.median(np.diff(dates[0].asi8))))
        if step <= pd.Timedelta(0):
            raise ValueError(f"The sampling interval must be positive: {step}")
        t0 = dates[0][0]
        out = []
        for w, d in enumerate(dates):
            if not np.all(np.diff(d.asi8) == step.value):
                raise ValueError(f"Window {w} has a nonuniform date axis")
            enc_len = len(d) - horizon
            if enc_len < 1:
                raise ValueError(f"Window {w} length {len(d)} <= horizon {horizon}")
            g = (d[enc_len] - t0) / step
            gi = int(round(g))
            if abs(float(g) - gi) > 1e-9:
                raise ValueError(f"Window {w} has timestamps outside the sampling grid")
            out.append(gi)
        return np.asarray(out)

    @staticmethod
    def verified_upto(origins: np.ndarray, horizon: int, i: int):
        return np.where(origins + horizon <= origins[i])[0]

    @staticmethod
    def causal_prefixes(origins: np.ndarray, horizon: int, i: int, l_min: int):
        L = np.minimum(origins[i] - origins, horizon)
        js = np.where((np.arange(len(origins)) < i) & (L >= l_min))[0]
        return [(int(j), int(L[j])) for j in js]


    def run(self, ckpt_paths, training_template, train_data_list, apply_data_list,
            target, known_reals, time_axis, norm_stats, model_class=None):
        import utils
        from pytorch_forecasting import TimeSeriesDataSet

        cfg = self.config
        if len(ckpt_paths) < 1:
            raise ValueError("At least one checkpoint is required")
        n_apply = len(apply_data_list)
        horizon = training_template.max_prediction_length
        origins = self.origin_step_indices(apply_data_list, horizon)
        if not np.all(np.diff(origins) > 0):
            raise ValueError("apply_data_list must be ordered by strictly increasing forecast origin")

        cls = model_class if model_class is not None else TemporalFusionTransformerNew
        models = [cls.load_from_checkpoint(p, weights_only=False)
                  for p in ckpt_paths]

        n_train = len(train_data_list)
        train_gids = [seg["group_id"] for seg in train_data_list]

        def _with_weight(seg, L):
            full = len(seg["date"])
            w = np.ones(full)
            w[full - horizon + L:] = 0.0
            return dict(seg, ft_weight=w)

        import time as _time

        refit_log = []
        results_ens = [None] * n_apply
        pool_sig_prev = None
        replay_capped = False
        n_batches = (n_apply + cfg.refit_every - 1) // cfg.refit_every
        batch_times = []
        for bi, i0 in enumerate(range(0, n_apply, cfg.refit_every)):
            t_batch = _time.time()
            i1 = min(i0 + cfg.refit_every, n_apply)
            if cfg.prefix_finetune:
                pairs = self.causal_prefixes(origins, horizon, i0, cfg.l_min_prefix)
                if len(pairs) > n_train:
                    pairs = pairs[-n_train:]
                    replay_capped = True
                pool = [_with_weight(apply_data_list[j], L) for j, L in pairs]
                pool_sig = tuple((j, L) for j, L in pairs)
            else:
                verified = self.verified_upto(origins, horizon, i0)
                v = verified
                if v.size > n_train:
                    v = v[-n_train:]
                    replay_capped = True
                pool = [apply_data_list[j] for j in v]
                pool_sig = tuple(int(j) for j in v)

            did_finetune = self._gate(pool, n_train) and pool_sig != pool_sig_prev
            if did_finetune:
                ft_train = list(train_data_list[len(pool):])
                if cfg.prefix_finetune:
                    ft_train = [
                        dict(seg, ft_weight=np.ones(len(seg["date"])))
                        for seg in ft_train
                    ]
                ft_list = ft_train + [
                    dict(seg, group_id=train_gids[k]) for k, seg in enumerate(pool)
                ]
                if cfg.prefix_finetune:
                    df_ft = utils.lists_to_df(
                        ft_list, target, known_reals,
                        explode_cols=["time_idx", "date", "ft_weight"])
                    ds_ft = TimeSeriesDataSet.from_dataset(
                        training_template, df_ft, stop_randomization=True,
                        weight="ft_weight")
                else:
                    df_ft = utils.lists_to_df(ft_list, target, known_reals)
                    ds_ft = TimeSeriesDataSet.from_dataset(
                        training_template, df_ft, stop_randomization=True)
                dl_ft = ds_ft.to_dataloader(
                    train=True, batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers)
                for m, model in enumerate(models):
                    t_ft = _time.time()
                    torch.manual_seed(cfg.seed + 977 * m + i0)
                    model.finetune_on(dl_ft, lr=cfg.ft_lr, max_steps=cfg.ft_steps,
                                      accelerator=cfg.accelerator)
                    if cfg.verbose:
                        print(f"[roll] batch {bi+1}/{n_batches} w[{i0},{i1}) "
                              f"pool={len(pool)} windows; ft model {m+1}/{len(models)} "
                              f"{_time.time()-t_ft:.0f}s", flush=True)
                pool_sig_prev = pool_sig
            batch = [apply_data_list[j] for j in range(i0, i1)]
            df_b = utils.lists_to_df(batch, target, known_reals)
            ds_b = TimeSeriesDataSet.from_dataset(
                training_template, df_b, predict=True, stop_randomization=True)
            dl_b = ds_b.to_dataloader(
                train=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
            per_model = [
                utils.predict_and_collect_results(
                    model, dl_b, df_b, ds_b, time_axis, norm_stats)
                for model in models
            ]
            for k in range(i1 - i0):
                rd = dict(per_model[0][k])
                rd["prediction"] = np.mean(
                    [np.asarray(pm[k]["prediction"], float) for pm in per_model], axis=0)
                results_ens[i0 + k] = rd
            refit_log.append({
                "batch": [int(i0), int(i1)],
                "n_pool": len(pool),
                "pool_steps": int(sum(L for _, L in pairs)) if cfg.prefix_finetune
                              else len(pool) * int(horizon),
                "finetuned": did_finetune,
            })
            batch_times.append(_time.time() - t_batch)
            if cfg.verbose:
                eta = np.mean(batch_times[-5:]) * (n_batches - bi - 1)
                print(f"[roll] batch {bi+1}/{n_batches} done "
                      f"({batch_times[-1]:.0f}s, ft={'yes' if did_finetune else 'no'}) "
                      f"| ETA ~{eta/60:.0f} min", flush=True)

        self.info_ = {
            "config": {k: getattr(cfg, k) for k in
                       ("refit_every", "ft_lr", "ft_steps", "alpha",
                        "prefix_finetune", "l_min_prefix",
                        "batch_size", "seed")},
            "ckpt_paths": [str(p) for p in ckpt_paths],
            "n_models": len(models),
            "n_apply": n_apply,
            "horizon": int(horizon),
            "replay_capped": replay_capped,
            "refit_log": refit_log,
        }
        return results_ens

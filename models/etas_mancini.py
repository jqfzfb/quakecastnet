import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve


class InjectionDrivenETASRollingModel:

    def __init__(
        self,
        dt=1.0,
        c=0.006,
        window_size=730,
        step_size=30
    ):
        self.dt = dt
        self.c = c
        self.window_size = window_size
        self.step_size = step_size

        self.param_series = None

    def _neg_log_likelihood(self, theta, rate_hist, inj_hist, start):
        cf, K, p = theta

        T = len(rate_hist)
        eps = 1e-8

        k = np.arange(T)
        kernel = K * (k * self.dt + self.c) ** (-p)
        kernel[0] = 0.0

        triggered = fftconvolve(rate_hist, kernel, mode="full")[:T]
        lambda_t = cf * inj_hist + triggered

        lam_w = np.clip(lambda_t[start:], eps, None)
        r_w = rate_hist[start:]

        return np.sum(lam_w - r_w * np.log(lam_w))

    def _fit_single_window(
        self,
        rate_hist,
        inj_hist,
        start,
        initial_params,
        bounds=((0.0, None), (1e-6, None), (1.2, 3.0))
    ):
        res = minimize(
            self._neg_log_likelihood,
            x0=np.array(initial_params),
            args=(rate_hist, inj_hist, start),
            bounds=bounds,
            method="L-BFGS-B"
        )

        cf, K, p = res.x

        return {
            "cf": cf,
            "K": K,
            "p": p,
            "c": self.c,
            "success": res.success,
            "loglik": -res.fun
        }

    def fit(self, rate_hist, inj_hist):
        rate_hist = np.asarray(rate_hist, dtype=float)
        inj_hist = np.asarray(inj_hist, dtype=float)
        T = len(rate_hist)

        assert len(inj_hist) == T, "Injection rate length mismatch"

        param_series = []
        prev_params = None

        for start in range(0, T - self.window_size + 1, self.step_size):
            end = start + self.window_size

            if prev_params is None:
                cf0 = 0.5 * rate_hist[start:end].mean() / max(inj_hist[start:end].mean(), 1e-8)
                init = (max(cf0, 1e-6), 0.5, 1.5)
            else:
                init = (
                    prev_params["cf"],
                    prev_params["K"],
                    prev_params["p"]
                )

            params = self._fit_single_window(
                rate_hist[:end],
                inj_hist[:end],
                start,
                initial_params=init
            )

            params["t_start"] = start
            params["t_end"] = end
            params["t_center"] = (start + end) // 2

            param_series.append(params)
            prev_params = params

        self.param_series = param_series
        return param_series

    def reconstruct(self, rate_hist, inj_hist):

        if self.param_series is None:
            raise RuntimeError("Model must be fitted before reconstruction.")

        rate_hist = np.asarray(rate_hist, dtype=float)
        inj_hist = np.asarray(inj_hist, dtype=float)
        T = len(rate_hist)

        assert len(inj_hist) == T, "Injection rate length mismatch"

        lambda_hat = np.zeros(T)

        for params in self.param_series:
            cf = params["cf"]
            K = params["K"]
            p = params["p"]
            c = params["c"]

            s, e = params["t_start"], params["t_end"]
            rate_window = rate_hist[s:e]
            inj_window = inj_hist[s:e]
            Tw = e - s

            k = np.arange(Tw)
            kernel = K * (k * self.dt + c) ** (-p)
            kernel[0] = 0.0
            triggered = fftconvolve(rate_window, kernel, mode="full")[:Tw]

            lambda_hat[s:e] = cf * inj_window + triggered

        last = self.param_series[-1]
        s = last["t_end"]
        if s < T:
            cf, K, p, c = last["cf"], last["K"], last["p"], last["c"]
            k = np.arange(T - s)
            kernel = K * (k * self.dt + c) ** (-p)
            kernel[0] = 0.0
            triggered = fftconvolve(rate_hist[s:], kernel, mode="full")[:T - s]
            lambda_hat[s:] = cf * inj_hist[s:] + triggered

        return lambda_hat

    def get_parameter_series(self):
        if self.param_series is None:
            raise RuntimeError("Model has not been fitted.")

        return {
            "t_center": np.array([p["t_center"] for p in self.param_series]),
            "cf": np.array([p["cf"] for p in self.param_series]),
            "K": np.array([p["K"] for p in self.param_series]),
            "p": np.array([p["p"] for p in self.param_series]),
        }

    def forecast(self, rate_hist, inj_hist, inj_future, n_future=None, refit=True):

        if self.param_series is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        rate_hist = list(np.asarray(rate_hist, dtype=float))
        inj_hist = list(np.asarray(inj_hist, dtype=float))
        inj_future = np.asarray(inj_future, dtype=float)

        if n_future is None:
            n_future = len(inj_future)
        assert len(inj_future) >= n_future, "inj_future shorter than n_future"

        future_pred = []

        prev_params = self.param_series[-1]

        for step in range(n_future):

            if refit:
                init = (
                    prev_params["cf"],
                    prev_params["K"],
                    prev_params["p"]
                )

                end = len(rate_hist)
                start = max(0, end - self.window_size)

                new_params = self._fit_single_window(
                    np.asarray(rate_hist),
                    np.asarray(inj_hist),
                    start,
                    initial_params=init
                )

                prev_params = new_params
            else:
                new_params = prev_params

            cf = new_params["cf"]
            K = new_params["K"]
            p = new_params["p"]
            c = new_params["c"]

            lam = cf * inj_future[step]
            for k, r in enumerate(reversed(rate_hist)):
                lag = (k + 1) * self.dt
                lam += K * r * (lag + c) ** (-p)

            lam = max(lam, 0.0)

            future_pred.append(lam)
            rate_hist.append(lam)
            inj_hist.append(inj_future[step])

        return np.array(future_pred)

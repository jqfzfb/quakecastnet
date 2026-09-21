import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

def fit_etas_rate_model(
    rate_hist,
    dt=1.0,
    initial_params=(0.1, 0.5, 0.01, 1.1),
    bounds=((1e-6, None), (1e-6, None), (1e-4, 10.0), (1.01, 3.0))
):

    rate_hist = np.asarray(rate_hist)
    T = len(rate_hist)
    eps = 1e-8

    def neg_log_likelihood(theta):
        mu, K, c, p = theta

        k = np.arange(1, T + 1)
        kernel = K * (k * dt + c) ** (-p)

        triggered = fftconvolve(rate_hist, kernel, mode="full")[:T]
        lambda_t = mu + triggered
        lambda_t = np.clip(lambda_t, eps, None)

        return np.sum(lambda_t - rate_hist * np.log(lambda_t))

    res = minimize(
        neg_log_likelihood,
        x0=np.array(initial_params),
        bounds=bounds,
        method="L-BFGS-B"
    )

    mu, K, c, p = res.x

    return {
        "mu": mu,
        "K": K,
        "c": c,
        "p": p,
        "success": res.success,
        "loglik": -res.fun
    }


def forecast_etas_rate(
    rate_hist,
    etas_params,
    n_forecast=14,
    dt=1.0
):

    mu = etas_params["mu"]
    K = etas_params["K"]
    c = etas_params["c"]
    p = etas_params["p"]

    rate_hist = list(rate_hist)
    rate_forecast = []

    for t in range(n_forecast):
        lam = mu
        for k, r in enumerate(reversed(rate_hist)):
            lag = (k + 1) * dt
            lam += K * r * (lag + c) ** (-p)

        rate_forecast.append(lam)
        rate_hist.append(lam)

    return np.array(rate_forecast)


def reconstruct_etas_rate(
    rate_hist,
    etas_params,
    dt=1.0
):

    rate_hist = np.asarray(rate_hist)
    T = len(rate_hist)

    mu = etas_params["mu"]
    K = etas_params["K"]
    c = etas_params["c"]
    p = etas_params["p"]

    k = np.arange(1, T + 1)
    kernel = K * (k * dt + c) ** (-p)

    triggered = fftconvolve(rate_hist, kernel, mode="full")[:T]
    background = np.full(T, mu)

    lambda_hat = background + triggered

    return lambda_hat, {
        "background": background,
        "triggered": triggered
    }


def fit_etas_injection_rate_model(
    rate_hist,
    injection_rate,
    dt=1.0,
    initial_params=(0.05, 0.05, 0.5, 0.01, 1.1),
    bounds=(
        (1e-6, None),
        (0.0, None),
        (1e-6, None),
        (1e-4, 10.0),
        (1.01, 3.0)
    )
):

    rate_hist = np.asarray(rate_hist)
    injection_rate = np.asarray(injection_rate)
    T = len(rate_hist)
    eps = 1e-8

    assert len(injection_rate) == T, "Injection rate length mismatch"

    def neg_log_likelihood(theta):
        mu0, beta, K, c, p = theta

        mu_t = mu0 + beta * injection_rate
        mu_t = np.clip(mu_t, eps, None)

        k = np.arange(1, T + 1)
        kernel = K * (k * dt + c) ** (-p)

        triggered = fftconvolve(rate_hist, kernel, mode="full")[:T]

        lambda_t = mu_t + triggered
        lambda_t = np.clip(lambda_t, eps, None)

        return np.sum(lambda_t - rate_hist * np.log(lambda_t))

    res = minimize(
        neg_log_likelihood,
        x0=np.array(initial_params),
        bounds=bounds,
        method="L-BFGS-B"
    )

    mu0, beta, K, c, p = res.x

    return {
        "mu0": mu0,
        "beta": beta,
        "K": K,
        "c": c,
        "p": p,
        "success": res.success,
        "loglik": -res.fun
    }


def reconstruct_etas_injection_rate(
    rate_hist,
    injection_rate,
    etas_params,
    dt=1.0
):

    rate_hist = np.asarray(rate_hist)
    injection_rate = np.asarray(injection_rate)
    T = len(rate_hist)

    mu0 = etas_params["mu0"]
    beta = etas_params["beta"]
    K = etas_params["K"]
    c = etas_params["c"]
    p = etas_params["p"]

    background = mu0 + beta * injection_rate

    k = np.arange(1, T + 1)
    kernel = K * (k * dt + c) ** (-p)

    triggered = fftconvolve(rate_hist, kernel, mode="full")[:T]

    lambda_hat = background + triggered

    return lambda_hat, {
        "background": background,
        "triggered": triggered
    }


def forecast_etas_injection_rate(
    rate_hist,
    injection_future,
    etas_params,
    n_forecast=None,
    dt=1.0
):

    mu0 = etas_params["mu0"]
    beta = etas_params["beta"]
    K = etas_params["K"]
    c = etas_params["c"]
    p = etas_params["p"]

    rate_hist = list(rate_hist)
    injection_future = np.asarray(injection_future)

    if n_forecast is None:
        n_forecast = len(injection_future)

    rate_forecast = []

    for t in range(n_forecast):
        mu_t = mu0 + beta * injection_future[t]
        lam = mu_t

        for k, r in enumerate(reversed(rate_hist)):
            lag = (k + 1) * dt
            lam += K * r * (lag + c) ** (-p)

        rate_forecast.append(lam)
        rate_hist.append(lam)

    return np.array(rate_forecast)


class ETASRollingModel:

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

    def _neg_log_likelihood(self, theta, rate_hist):
        mu, K, p = theta

        T = len(rate_hist)
        eps = 1e-8

        k = np.arange(1, T + 1)
        kernel = K * (k * self.dt + self.c) ** (-p)

        triggered = fftconvolve(rate_hist, kernel, mode="full")[:T]
        lambda_t = mu + triggered
        lambda_t = np.clip(lambda_t, eps, None)

        return np.sum(lambda_t - rate_hist * np.log(lambda_t))

    def _fit_single_window(
        self,
        rate_window,
        initial_params,
        bounds=((1e-6, None), (1e-6, None), (1.2, 3.0))
    ):
        res = minimize(
            self._neg_log_likelihood,
            x0=np.array(initial_params),
            args=(rate_window,),
            bounds=bounds,
            method="L-BFGS-B"
        )

        mu, K, p = res.x

        return {
            "mu": mu,
            "K": K,
            "p": p,
            "c": self.c,
            "success": res.success,
            "loglik": -res.fun
        }


    def fit(self, rate_hist):
        rate_hist = np.asarray(rate_hist)
        T = len(rate_hist)

        param_series = []
        prev_params = None

        for start in range(0, T - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window_rate = rate_hist[start:end]

            if prev_params is None:
                init = (0.1, 0.5, 1.5)
            else:
                init = (
                    prev_params["mu"],
                    prev_params["K"],
                    prev_params["p"]
                )

            params = self._fit_single_window(
                window_rate,
                initial_params=init
            )

            params["t_start"] = start
            params["t_end"] = end
            params["t_center"] = (start + end) // 2

            param_series.append(params)
            prev_params = params

        self.param_series = param_series
        return param_series


    def reconstruct(self, rate_hist):

        if self.param_series is None:
            raise RuntimeError("Model must be fitted before reconstruction.")

        rate_hist = np.asarray(rate_hist)
        T = len(rate_hist)

        lambda_hat = np.zeros(T)

        for params in self.param_series:
            mu = params["mu"]
            K = params["K"]
            p = params["p"]
            c = params["c"]

            t0 = params["t_start"]
            t1 = params["t_end"]

            for t in range(t0, min(t1, T)):
                lam = mu
                for k in range(t):
                    lag = (t - k) * self.dt
                    lam += K * rate_hist[k] * (lag + c) ** (-p)
                lambda_hat[t] = lam

        last_params = self.param_series[-1]
        mu = last_params["mu"]
        K = last_params["K"]
        p = last_params["p"]
        c = last_params["c"]

        last_covered_t = self.param_series[-1]["t_end"]

        for t in range(last_covered_t, T):
            lam = mu
            for k in range(t):
                lag = (t - k) * self.dt
                lam += K * rate_hist[k] * (lag + c) ** (-p)
            lambda_hat[t] = lam

        return lambda_hat

    def predict_historical(self, rate_hist, n_init=1):

        if self.param_series is None:
            raise RuntimeError("Model must be fitted before prediction.")

        rate_hist = np.asarray(rate_hist)
        T = len(rate_hist)

        lambda_pred = np.zeros(T)

        lambda_pred[:n_init] = rate_hist[:n_init]

        def get_params_at_time(t):
            for params in self.param_series:
                if params["t_start"] <= t < params["t_end"]:
                    return params
            return self.param_series[-1]

        for t in range(n_init, T):
            params = get_params_at_time(t)

            mu = params["mu"]
            K  = params["K"]
            p  = params["p"]
            c  = params["c"]

            lam = mu
            for k in range(t):
                lag = (t - k) * self.dt
                lam += K * lambda_pred[k] * (lag + c) ** (-p)

            lambda_pred[t] = lam

        return lambda_pred

    def get_parameter_series(self):
        if self.param_series is None:
            raise RuntimeError("Model has not been fitted.")

        return {
            "t_center": np.array([p["t_center"] for p in self.param_series]),
            "mu": np.array([p["mu"] for p in self.param_series]),
            "K": np.array([p["K"] for p in self.param_series]),
            "p": np.array([p["p"] for p in self.param_series]),
        }


    def forecast(self, rate_hist, n_future=14, refit=True):

        if self.param_series is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        rate_hist = list(np.asarray(rate_hist))

        future_pred = []

        prev_params = self.param_series[-1]

        for step in range(n_future):

            if refit:
                if len(rate_hist) < self.window_size:
                    window_rate = rate_hist
                else:
                    window_rate = rate_hist[-self.window_size:]

                init = (
                    prev_params["mu"],
                    prev_params["K"],
                    prev_params["p"]
                )

                new_params = self._fit_single_window(
                    window_rate,
                    initial_params=init
                )

                prev_params = new_params
            else:
                new_params = prev_params

            mu = new_params["mu"]
            K  = new_params["K"]
            p  = new_params["p"]
            c  = new_params["c"]

            lam = mu
            for k, r in enumerate(reversed(rate_hist)):
                lag = (k + 1) * self.dt
                lam += K * r * (lag + c) ** (-p)

            future_pred.append(lam)
            rate_hist.append(lam)

        return np.array(future_pred)


    def forecast_one_step_ahead(
        self,
        rate_hist,
        n_future=14
    ):

        rate_hist = np.asarray(rate_hist)
        T = len(rate_hist)

        future_pred = []
        future_param_series = []

        for step in range(n_future):

            t_current = T - n_future + step

            if t_current < self.window_size:
                window_rate = rate_hist[:t_current]
                t_start = 0
            else:
                t_start = t_current - self.window_size
                window_rate = rate_hist[t_start:t_current]

            t_end = t_current
            t_center = (t_start + t_end) // 2

            if len(window_rate) < 5:
                future_pred.append(np.nan)
                future_param_series.append(None)
                continue

            init = (0.1, 0.5, 1.5)

            params = self._fit_single_window(
                window_rate,
                initial_params=init
            )

            params["t_start"] = t_start
            params["t_end"] = t_end
            params["t_center"] = t_center

            future_param_series.append(params)

            mu = params["mu"]
            K  = params["K"]
            p  = params["p"]
            c  = params["c"]

            lam = mu

            for k, r in enumerate(reversed(rate_hist[:t_current])):
                lag = (k + 1) * self.dt
                lam += K * r * (lag + c) ** (-p)

            future_pred.append(lam)

        return np.array(future_pred), future_param_series
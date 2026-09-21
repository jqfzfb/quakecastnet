import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve


class ConvolutionalInjectionModel:

    def __init__(self, dt=1.0, q=2.0, fit_q=False):
        self.dt = dt
        self.q = q
        self.fit_q = fit_q

        self.params = None

    def _kernel_weights(self, t_r, q, T):
        k = np.arange(T)
        w = (self.dt / t_r) / (1.0 + k * self.dt / t_r) ** q
        return w

    def _convolve(self, inj, t_r, q):
        w = self._kernel_weights(t_r, q, len(inj))
        return fftconvolve(inj, w, mode="full")[:len(inj)]

    def fit(self, rate_hist, inj_hist):
        rate_hist = np.asarray(rate_hist, dtype=float)
        inj_hist = np.asarray(inj_hist, dtype=float)
        T = len(rate_hist)

        assert len(inj_hist) == T, "Injection rate length mismatch"

        eps = 1e-8

        R0_init = max(rate_hist.sum(), eps) / max(inj_hist.sum(), eps)
        t_r_init = 5.0 * self.dt

        def unpack(theta):
            log_R0, log_tr = theta[0], theta[1]
            q = np.exp(theta[2]) if self.fit_q else self.q
            return np.exp(log_R0), np.exp(log_tr), q

        def neg_log_likelihood(theta):
            R0, t_r, q = unpack(theta)
            lambda_t = R0 * self._convolve(inj_hist, t_r, q)
            lambda_t = np.clip(lambda_t, eps, None)
            return np.sum(lambda_t - rate_hist * np.log(lambda_t))

        theta0 = [np.log(R0_init), np.log(t_r_init)]
        if self.fit_q:
            theta0.append(np.log(self.q))

        res = minimize(neg_log_likelihood, x0=np.array(theta0),
                       method="Nelder-Mead")

        R0, t_r, q = unpack(res.x)

        self.params = {
            "R0": R0,
            "t_r": t_r,
            "q": q,
            "success": res.success,
            "loglik": -res.fun,
        }
        return self.params

    def get_params(self):
        if self.params is None:
            raise RuntimeError("Model has not been fitted.")
        return dict(self.params)

    def reconstruct(self, inj_hist):
        if self.params is None:
            raise RuntimeError("Model must be fitted before reconstruction.")

        inj_hist = np.asarray(inj_hist, dtype=float)
        lam = self.params["R0"] * self._convolve(
            inj_hist, self.params["t_r"], self.params["q"])
        return np.clip(lam, 0.0, None)

    def forecast(self, inj_hist, inj_future, n_future=None):
        if self.params is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        inj_hist = np.asarray(inj_hist, dtype=float)
        inj_future = np.asarray(inj_future, dtype=float)

        if n_future is None:
            n_future = len(inj_future)
        assert len(inj_future) >= n_future, "inj_future shorter than n_future"

        inj_full = np.concatenate([inj_hist, inj_future[:n_future]])
        lam_full = self.params["R0"] * self._convolve(
            inj_full, self.params["t_r"], self.params["q"])

        return np.clip(lam_full[-n_future:], 0.0, None)

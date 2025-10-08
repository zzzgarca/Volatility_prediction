import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

class Metrics:
    @staticmethod
    def evaluate(
        y_true,
        y_pred,
        horizons=[1, 3, 5, 10, 20, -1],
        epsilon=1e-12,
        qlike_mode="ratio",
        qlike_floor="auto",
        qlike_calibrate=True
    ):
        results = {}
        assert y_true.ndim == 2 and y_pred.ndim == 2
        assert y_true.shape == y_pred.shape
        T = y_true.shape[1]

        def _auto_floor(arr):
            a = np.asarray(arr, dtype=np.float64).ravel()
            m = np.isfinite(a) & (a > 0.0)
            if not np.any(m):
                return float(epsilon)
            base = np.percentile(a[m], 5.0)
            return float(max(epsilon, 1e-6 * base))

        for h in horizons:
            label = f"{h} day(s)" if h > 0 else "full horizon"
            h_slice = h if h > 0 else T
            if h_slice > T:
                continue

            yt = y_true[:, :h_slice].astype(np.float64).ravel()
            yp = y_pred[:, :h_slice].astype(np.float64).ravel()
            mask = np.isfinite(yt) & np.isfinite(yp)
            yt = yt[mask]
            yp = yp[mask]

            if yt.size == 0:
                results[f"{label} MAE"] = np.nan
                results[f"{label} RMSE"] = np.nan
                results[f"{label} R2"] = np.nan
                results[f"{label} Pearson r"] = np.nan
                results[f"{label} QLIKE"] = np.nan
                continue

            mae = float(mean_absolute_error(yt, yp))
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            r2 = float(r2_score(yt, yp))

            ytm = yt - yt.mean()
            ypm = yp - yp.mean()
            denom = np.sqrt((ytm**2).mean() * (ypm**2).mean())
            pearson_r = float((ytm * ypm).mean() / denom) if denom > epsilon else np.nan

            if qlike_floor == "auto":
                floor = _auto_floor(yt)
            else:
                floor = max(float(qlike_floor), float(epsilon))

            yt_pos = np.clip(yt, floor, None)
            yp_pos = np.clip(yp, floor, None)

            if qlike_calibrate:
                c = float(np.mean(yt_pos / yp_pos))
                yp_pos = np.clip(yp_pos * c, floor, None)

            if qlike_mode == "ratio":
                r = yt_pos / yp_pos
                qlike = float(np.mean(r - np.log(r) - 1.0))
            elif qlike_mode == "log":
                qlike = float(np.mean(np.log(yp_pos) + (yt_pos / yp_pos)))
            else:
                raise ValueError("qlike_mode must be 'ratio' or 'log'")

            results[f"{label} MAE"] = mae
            results[f"{label} RMSE"] = rmse
            results[f"{label} R2"] = r2
            results[f"{label} Pearson r"] = pearson_r
            results[f"{label} QLIKE"] = qlike

        return results

    @staticmethod
    def to_frame(results_dict):
        keys = sorted(results_dict.keys())
        df = pd.DataFrame({"metric": keys, "value": [results_dict[k] for k in keys]})
        return df

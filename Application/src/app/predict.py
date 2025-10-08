import os
import numpy as np
import pandas as pd
import joblib
from data_utils import DataUtils

class Predict:
    def __init__(self, bundle=None):
        self.wrapper = None
        self.model_kind = None
        self.scaler_X = None
        self.y_ctx = None
        self.shapes = None
        if bundle is not None:
            self._load_bundle_obj(bundle)

    @classmethod
    def load(cls, bundle_path):
        bundle = joblib.load(bundle_path)
        return cls(bundle)

    def _load_bundle_obj(self, b):
        self.wrapper = b["wrapper"]
        self.model_kind = b["model_kind"]
        self.scaler_X = b.get("scaler_X", None)
        self.y_ctx = b.get("y_ctx", None)
        self.shapes = b.get("shapes", {})

    def _inverse_y(self, y):
        if self.y_ctx and self.y_ctx.get("mode") == "log":
            return np.exp(y)
        return y

    def predict_next(self, last_window_returns, last_window_garch=None):
        L = int(self.shapes["input_len"])
        F = int(self.shapes["features"])
        r = np.asarray(last_window_returns, dtype=float).reshape(-1)
        if r.shape[0] != L:
            raise ValueError("last_window_returns length must equal input_len")
        if F == 2:
            if last_window_garch is None:
                raise ValueError("last_window_garch is required for 2-feature models")
            g = np.asarray(last_window_garch, dtype=float).reshape(-1)
            if g.shape[0] != L:
                raise ValueError("last_window_garch length must equal input_len")
            X = np.stack([r, g], axis=-1)[None, ...]
        else:
            X = r[:, None][None, ...]
        if self.scaler_X is not None:
            B, T, F = X.shape
            Xs = self.scaler_X.transform(X.reshape(-1, F)).reshape(B, T, F)
        else:
            Xs = X
        if self.model_kind in ("kan", "kan_hier"):
            Xf = Xs.reshape(1, -1)
            yp = self.wrapper.predict(Xf)
            Tpred = int(self.shapes["pred_len"])
            yp = yp.reshape(1, Tpred)
        else:
            yp = self.wrapper.predict(Xs)
            yp = yp[:, :, 0]
        return self._inverse_y(yp)

    def predict_batch(self, X):
        X = np.asarray(X, dtype=float)
        if self.model_kind in ("kan", "kan_hier"):
            if X.ndim == 3:
                B, T, F = X.shape
                if self.scaler_X is not None:
                    Xs = self.scaler_X.transform(X.reshape(-1, F)).reshape(B, T, F)
                else:
                    Xs = X
                Xf = Xs.reshape(B, -1)
            elif X.ndim == 2:
                Xf = X
            else:
                raise ValueError("X must be 2D flattened or 3D (batch,time,features)")
            yp = self.wrapper.predict(Xf)
            Tpred = int(self.shapes["pred_len"])
            yp = yp.reshape(-1, Tpred)
        else:
            if X.ndim != 3:
                raise ValueError("For kan_lstm, X must be 3D (batch,time,features)")
            B, T, F = X.shape
            if self.scaler_X is not None:
                Xs = self.scaler_X.transform(X.reshape(-1, F)).reshape(B, T, F)
            else:
                Xs = X
            yp = self.wrapper.predict(Xs)
            yp = yp[:, :, 0]
        return self._inverse_y(yp)

    def predict_from_csv(self, csv_path, garch_window=None, use_garch=True, input_len=None, save_csv_path=None):
        r_log, r_var = DataUtils.load_two_row_csv(csv_path)
        L = int(self.shapes["input_len"]) if input_len is None else int(input_len)
        if use_garch:
            g = DataUtils.compute_garch_rolling(r_log, start_index=max(100, L), window=garch_window)
        else:
            g = None
        if len(r_log) < L + 1:
            raise ValueError("Series too short to extract last window")
        last_r = r_log[-L:]
        last_g = None
        if g is not None and np.isfinite(g[-L:]).all():
            last_g = g[-L:]
        ypred = self.predict_next(last_r, last_g)
        df = pd.DataFrame(ypred, columns=[f"H{h+1}" for h in range(ypred.shape[1])])
        if save_csv_path:
            os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
            df.to_csv(save_csv_path, index=False)
        return df

    def to_dataframe(self, y_pred, index=None, cols=None):
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            y_pred = y_pred[None, :]
        if cols is None:
            cols = [f"H{h+1}" for h in range(y_pred.shape[1])]
        return pd.DataFrame(y_pred, index=index, columns=cols)

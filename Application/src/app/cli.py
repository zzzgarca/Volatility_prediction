import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Dict, Any, Optional, Callable
from model_wrapper import Wrapper
from viz import Vizualize
from metrics import Metrics
from data_utils import DataUtils

class Command:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_csv_two_series(self, path: str, log_col: Optional[str] = None, rv_col: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        return DataUtils.load_two_series_csv(path, log_col=log_col, rv_col=rv_col, orientation="auto")

    def _fit_wrapper(self, arch: str, X_tr: np.ndarray, Y_tr: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray, params: Dict[str, Any], log_fn: Optional[Callable[[str], None]] = None) -> Wrapper:
        p = dict(params)
        no_tasks = int(p.pop("no_tasks", 1))
        if arch in ("kan", "kan_hier"):
            X_tr_2d = X_tr.reshape(X_tr.shape[0], -1)
            X_val_2d = X_val.reshape(X_val.shape[0], -1)
            Y_tr_2d = Y_tr.reshape(Y_tr.shape[0], -1)
            Y_val_2d = Y_val.reshape(Y_val.shape[0], -1)
            w = Wrapper(arch=arch, input_dim=X_tr_2d.shape[1], output_dim=Y_tr_2d.shape[1], no_tasks=no_tasks, **p)
            w.fit(X_tr_2d, Y_tr_2d, X_val_2d, Y_val_2d, log_callback=log_fn)
            return w
        elif arch == "kan_lstm":
            w = Wrapper(arch=arch, input_dim=X_tr.shape[2], output_dim=Y_tr.shape[2], no_tasks=Y_tr.shape[2], pred_len=Y_tr.shape[1], **p)
            w.fit(X_tr, Y_tr, X_val, Y_val, log_callback=log_fn)
            return w
        else:
            raise ValueError("Unknown arch")

    def _predict_wrapper(self, w: Wrapper, arch: str, X_te: np.ndarray, horizon: int, no_tasks: int) -> np.ndarray:
        if arch in ("kan", "kan_hier"):
            X_te_2d = X_te.reshape(X_te.shape[0], -1)
            pred = w.predict(X_te_2d)
            pred = pred.reshape(pred.shape[0], horizon, no_tasks)
            y1 = pred[:, 0, 0]
            return y1
        else:
            pred = w.predict(X_te)
            y1 = pred[:, 0, 0]
            return y1

    def train(
        self,
        csv_path: str,
        arch: str = "kan",
        window: int = 100,
        horizon: int = 28,
        test_frac: float = 0.2,
        out_prefix: str = "run",
        log_col: Optional[str] = None,
        rv_col: Optional[str] = None,
        garch_save_path: Optional[str] = None,
        verbose: bool = True,
        **model_params
    ):
        lr, rv = self._load_csv_two_series(csv_path, log_col=log_col, rv_col=rv_col)
        X, Y_rv = DataUtils.make_windows_from_series(rv, win=window, horizon=horizon)
        if arch == "kan":
            Y = Y_rv
            no_tasks = 1
        elif arch in ("kan_hier", "kan_lstm"):
            g = DataUtils.compute_garch_rolling(lr, start_index=max(50, window), window=None, verbose=verbose, save_path=garch_save_path)
            Y_g = DataUtils.align_aux_for_horizon(g, win=window, horizon=horizon)
            n = min(Y_rv.shape[0], Y_g.shape[0])
            Y = np.concatenate([Y_rv[:n], Y_g[:n]], axis=2)
            X = X[:n]
            no_tasks = 2
        else:
            raise ValueError("Unknown arch")

        n = X.shape[0]
        k = max(1, min(n - 1, int(round((1 - float(test_frac)) * n))))
        X_tr, X_te = X[:k], X[k:]
        Y_tr, Y_te = Y[:k], Y[k:]

        print(f"[train] X_tr {X_tr.shape} Y_tr {Y_tr.shape} | X_te {X_te.shape} Y_te {Y_te.shape}")

        w = self._fit_wrapper(arch, X_tr, Y_tr, X_te, Y_te, {**model_params, "no_tasks": no_tasks})

        if arch in ("kan", "kan_hier"):
            X_te_2d = X_te.reshape(X_te.shape[0], -1)
            yp = w.predict(X_te_2d).reshape(X_te.shape[0], horizon, no_tasks)[:, :, 0]
        else:
            yp = w.predict(X_te)[:, :, 0]
        yt = Y_te[:, :, 0]

        metrics = Metrics.evaluate(yt, yp)

        save_dir = os.path.join(self.output_dir, out_prefix)
        os.makedirs(save_dir, exist_ok=True)
        mplot, cplot = Vizualize.generate_both(yt, yp, eval_results=metrics, save_dir=save_dir, basename=out_prefix)

        model_file = os.path.join(save_dir, f"{out_prefix}_model.pkl")
        joblib.dump({"wrapper": w, "arch": arch, "window": window, "horizon": horizon, "no_tasks": no_tasks}, model_file)

        metrics_file = os.path.join(save_dir, f"{out_prefix}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        return {
            "model_path": model_file,
            "metrics_path": mplot,
            "curves_path": cplot,
            "metrics": metrics,
            "y_true": yt,
            "y_pred": yp,
            "garch_csv": garch_save_path
        }
    
    


    def predict(self, csv_path: str, model_path: str, out_prefix: str = "predict", log_col: Optional[str] = None, rv_col: Optional[str] = None, outputs_dir: str = "outputs"):
        saved = joblib.load(model_path)
        w: Wrapper = saved["wrapper"]
        arch = saved["arch"]
        window = saved["window"]
        horizon = saved["horizon"]
        no_tasks = saved["no_tasks"]
        lr, rv = self._load_csv_two_series(csv_path, log_col=log_col, rv_col=rv_col)
        X_new, _ = DataUtils.make_windows_from_series(rv, win=window, horizon=horizon)
        y1 = self._predict_wrapper(w, arch, X_new, horizon, no_tasks)
        save_dir = os.path.join(outputs_dir, out_prefix)
        os.makedirs(save_dir, exist_ok=True)
        out_csv = os.path.join(save_dir, f"{out_prefix}_pred_h1.csv")
        pd.DataFrame({"pred_h1": y1}).to_csv(out_csv, index=False)
        return {"pred_csv": out_csv}

    def _parse_bool(x: str) -> bool:
        x = str(x).lower().strip()
        if x in ("1", "true", "yes", "y", "t"):
            return True
        if x in ("0", "false", "no", "n", "f"):
            return False
        raise argparse.ArgumentTypeError("Boolean expected")
    
    def main():
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="cmd", required=True)
    
        pt = sub.add_parser("train")
        pt.add_argument("--csv", required=True)
        pt.add_argument("--arch", default="kan", choices=["kan", "kan_hier", "kan_lstm"])
        pt.add_argument("--window", type=int, default=100)
        pt.add_argument("--horizon", type=int, default=28)
        pt.add_argument("--test_frac", type=float, default=0.2)
        pt.add_argument("--out_prefix", default="run")
        pt.add_argument("--output_dir", default="outputs")
        pt.add_argument("--epochs", type=int, default=50)
        pt.add_argument("--batch_size", type=int, default=512)
        pt.add_argument("--lr", type=float, default=1e-3)
        pt.add_argument("--hidden_dim", type=int, default=128)
        pt.add_argument("--hidden_layers", type=int, default=3)
        pt.add_argument("--dropout", type=float, default=0.0)
        pt.add_argument("--knots", type=int, default=8)
        pt.add_argument("--spline_power", type=int, default=3)
        pt.add_argument("--loss_type", default="gauss_nll_var")
        pt.add_argument("--target_is_logvar", type=lambda x: str(x).lower() in ("1","true","t","yes","y"), default=True)
        pt.add_argument("--patience", type=int, default=10)
        pt.add_argument("--min_epochs", type=int, default=20)
        pt.add_argument("--min_delta", type=float, default=1e-4)
        pt.add_argument("--warmup_aux_epochs", type=int, default=10)
        pt.add_argument("--joint_epochs", type=int, default=10)
        pt.add_argument("--log_col", default=None)
        pt.add_argument("--rv_col", default=None)
        pt.add_argument("--save_garch", type=lambda x: str(x).lower() in ("1","true","t","yes","y"), default=True)
        pt.add_argument("--garch_csv", default=None)
        pt.add_argument("--verbose", type=lambda x: str(x).lower() in ("1","true","t","yes","y"), default=True)
    
        pp = sub.add_parser("predict")
        pp.add_argument("--csv", required=True)
        pp.add_argument("--model_path", required=True)
        pp.add_argument("--out_prefix", default="predict")
        pp.add_argument("--output_dir", default="outputs")
        pp.add_argument("--log_col", default=None)
        pp.add_argument("--rv_col", default=None)
    
        args = p.parse_args()
        if args.cmd == "train":
            cmd = Command(output_dir=args.output_dir)
            g_path = args.garch_csv if args.save_garch else None
            model_params = dict(
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, dropout=args.dropout,
                knots=args.knots, spline_power=args.spline_power, loss_type=args.loss_type,
                target_is_logvar=args.target_is_logvar, patience=args.patience,
                min_epochs=args.min_epochs, min_delta=args.min_delta,
                warmup_aux_epochs=args.warmup_aux_epochs, joint_epochs=args.joint_epochs
            )
            cmd.train(
                csv_path=args.csv, arch=args.arch, window=args.window, horizon=args.horizon,
                test_frac=args.test_frac, out_prefix=args.out_prefix,
                log_col=args.log_col, rv_col=args.rv_col,
                garch_save_path=g_path, verbose=args.verbose, **model_params
            )
        else:
            cmd = Command(output_dir=args.output_dir)
            cmd.predict(
                csv_path=args.csv, model_path=args.model_path,
                out_prefix=args.out_prefix, outputs_dir=args.output_dir,
                log_col=args.log_col, rv_col=args.rv_col
            )


if __name__ == "__main__":
    main()

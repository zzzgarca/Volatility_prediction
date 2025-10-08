import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from metrics import Metrics

class Vizualize:
    @staticmethod
    def _ensure_2d(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 1:
            a = a[None, :]
        return a

    @staticmethod
    def _metric_file_name(basename: str, metric_key: str) -> str:
        s = metric_key.lower().replace(" ", "_").replace("/", "_")
        return f"{basename}_metric_{s}.png"

    @staticmethod
    def plot_metric_single(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_key: str,
        eval_results: Optional[Dict[str, float]] = None,
        horizons: List[int] = (1, 3, 5, 10, 20, -1),
        save_path: str = "plots/metric.png",
        dpi: int = 150,
        figsize: Tuple[int, int] = (8, 4),
    ) -> str:
        yt = Vizualize._ensure_2d(y_true)
        yp = Vizualize._ensure_2d(y_pred)
        res = eval_results if eval_results is not None else Metrics.evaluate(yt, yp, horizons=list(horizons))
        labels = [(f"{h} day(s)" if h > 0 else "full horizon") for h in horizons]
        max_pos = max([h for h in horizons if isinstance(h, int) and h > 0] + [0])
        xs = [h if (isinstance(h, int) and h > 0) else max_pos + 1 for h in horizons]
        xt = [str(h) if (isinstance(h, int) and h > 0) else "full" for h in horizons]
        ys = [res.get(f"{lab} {metric_key}", np.nan) for lab in labels]
        plt.figure(figsize=figsize)
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Horizon")
        plt.xticks(xs, xt)
        plt.ylabel(metric_key)
        plt.title(metric_key)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        return save_path

    @staticmethod
    def plot_curves_h1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "plots/actual_vs_predicted_h1.png",
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 4),
    ) -> str:
        yt = Vizualize._ensure_2d(y_true)
        yp = Vizualize._ensure_2d(y_pred)
        yth = yt[:, 0] if yt.shape[1] >= 1 else yt.squeeze(-1)
        yph = yp[:, 0] if yp.shape[1] >= 1 else yp.squeeze(-1)
        n = int(min(len(yth), len(yph)))
        yth = np.asarray(yth[:n], dtype=float)
        yph = np.asarray(yph[:n], dtype=float)
        x = np.arange(n)
        plt.figure(figsize=figsize)
        plt.plot(x, yth, label="Actual (H=1)")
        plt.plot(x, yph, label="Predicted (H=1)", linestyle="--")
        plt.xlabel("Sample index")
        plt.ylabel("Value")
        plt.title("Actual vs Predicted (H=1)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        return save_path

    @staticmethod
    def generate_both(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        eval_results: Optional[Dict[str, float]] = None,
        horizons: List[int] = (1, 3, 5, 10, 20, -1),
        save_dir: str = "plots",
        basename: str = "result",
    ) -> Tuple[List[str], str]:
        yt = Vizualize._ensure_2d(y_true)
        yp = Vizualize._ensure_2d(y_pred)
        res = eval_results if eval_results is not None else Metrics.evaluate(yt, yp, horizons=list(horizons))
        metric_keys = ["MAE", "RMSE", "R2", "Pearson r", "QLIKE"]
        metric_paths: List[str] = []
        for mk in metric_keys:
            mp = os.path.join(save_dir, Vizualize._metric_file_name(basename, mk))
            p = Vizualize.plot_metric_single(yt, yp, mk, eval_results=res, horizons=horizons, save_path=mp)
            metric_paths.append(p)
        curves_path = os.path.join(save_dir, f"{basename}_y_plot.png")
        cpath = Vizualize.plot_curves_h1(yt, yp, save_path=curves_path)
        return metric_paths, cpath

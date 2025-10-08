import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from cli import Command as KANCommand

class ImageCarousel(tk.Toplevel):
    def __init__(self, master, paths):
        super().__init__(master)
        self.title("Vizualizări")
        self.geometry("1000x650")
        self.paths = [p for p in paths if isinstance(p, str) and p and os.path.exists(p)]
        self.idx = 0
        self.img_label = tk.Label(self)
        self.img_label.pack(fill="both", expand=True)
        nav = ttk.Frame(self)
        nav.pack(side="bottom", fill="x")
        ttk.Button(nav, text="◀", command=self.prev_img).pack(side="left", padx=8, pady=6)
        ttk.Button(nav, text="▶", command=self.next_img).pack(side="right", padx=8, pady=6)
        self.bind("<Left>", lambda e: self.prev_img())
        self.bind("<Right>", lambda e: self.next_img())
        self.show()

    def _load_image(self, path, max_w, max_h):
        try:
            from PIL import Image, ImageTk
            img = Image.open(path)
            w, h = img.size
            scale = min(max_w / max(1, w), max_h / max(1, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            try:
                return tk.PhotoImage(file=path)
            except Exception:
                return None

    def show(self):
        if not self.paths:
            messagebox.showinfo("Vizualizări", "Nu există imagini de afișat.")
            self.destroy()
            return
        path = self.paths[self.idx]
        w = self.img_label.winfo_width() or 960
        h = self.img_label.winfo_height() or 560
        img = self._load_image(path, w, h)
        if img is None:
            self.img_label.config(text=os.path.basename(path))
        else:
            self.img_label.config(image=img, text="")
            self.img_label.image = img
        self.title(os.path.basename(path))

    def prev_img(self):
        if not self.paths:
            return
        self.idx = (self.idx - 1) % len(self.paths)
        self.show()

    def next_img(self):
        if not self.paths:
            return
        self.idx = (self.idx + 1) % len(self.paths)
        self.show()

class KANUI:
    def __init__(self, root=None):
        self.root = root or tk.Tk()
        self.root.title("KAN pentru serii de timp")
        self.root.geometry("1180x820")
        self.root.minsize(980, 680)
        self.cmd = KANCommand()

        self.csv_path = tk.StringVar(value="")
        self.model_path = tk.StringVar(value="")

        self.arch = tk.StringVar(value="kan")
        self.window = tk.StringVar(value="100")
        self.horizon = tk.StringVar(value="28")

        self.lr = tk.StringVar(value="0.001")
        self.weight_decay = tk.StringVar(value="0.0")
        self.batch_size = tk.StringVar(value="256")
        self.epochs = tk.StringVar(value="50")
        self.knots = tk.StringVar(value="8")
        self.spline_power = tk.StringVar(value="3")
        self.loss_type = tk.StringVar(value="gauss_nll_var")
        self.target_is_logvar = tk.BooleanVar(value=False)
        self.save_garch = tk.BooleanVar(value=True)

        self.models_dir = os.path.abspath("./artifacts/models")
        self.outputs_dir = os.path.abspath("./artifacts/outputs")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        self.last_plots = []
        self._build()

    def _build(self):
        pad = {"padx": 8, "pady": 6}

        frm_load = ttk.LabelFrame(self.root, text="Date")
        frm_load.grid(row=0, column=0, sticky="nsew", **pad)
        frm_load.columnconfigure(1, weight=1)
        ttk.Label(frm_load, text="Fișier CSV (două coloane: lr, rv)").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_load, textvariable=self.csv_path).grid(row=0, column=1, sticky="ew")
        ttk.Button(frm_load, text="Alege…", command=self._pick_csv).grid(row=0, column=2)
        ttk.Label(frm_load, text="Model salvat").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_load, textvariable=self.model_path).grid(row=1, column=1, sticky="ew")
        ttk.Button(frm_load, text="Încarcă…", command=self._pick_model).grid(row=1, column=2)

        frm_train = ttk.LabelFrame(self.root, text="Antrenare")
        frm_train.grid(row=1, column=0, sticky="nsew", **pad)
        for c in range(8):
            frm_train.columnconfigure(c, weight=1)

        ttk.Label(frm_train, text="Arhitectură").grid(row=0, column=0, sticky="w")
        ttk.Combobox(frm_train, textvariable=self.arch, values=["kan", "kan_hier", "kan_lstm"], state="readonly", width=16).grid(row=0, column=1, sticky="w")

        ttk.Label(frm_train, text="Fereastră de obs.").grid(row=0, column=2, sticky="w")
        ttk.Entry(frm_train, textvariable=self.window, width=12).grid(row=0, column=3, sticky="w")

        ttk.Label(frm_train, text="Orizont de predicție").grid(row=0, column=4, sticky="w")
        ttk.Entry(frm_train, textvariable=self.horizon, width=12).grid(row=0, column=5, sticky="w")

        ttk.Label(frm_train, text="Rată de învățare").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_train, textvariable=self.lr, width=16).grid(row=1, column=1, sticky="w")

        ttk.Label(frm_train, text="Penalizare L2").grid(row=1, column=2, sticky="w")
        ttk.Entry(frm_train, textvariable=self.weight_decay, width=16).grid(row=1, column=3, sticky="w")

        ttk.Label(frm_train, text="Dimensiune batch").grid(row=1, column=4, sticky="w")
        ttk.Entry(frm_train, textvariable=self.batch_size, width=16).grid(row=1, column=5, sticky="w")

        ttk.Label(frm_train, text="Număr de epoci").grid(row=1, column=6, sticky="w")
        ttk.Entry(frm_train, textvariable=self.epochs, width=16).grid(row=1, column=7, sticky="w")

        ttk.Label(frm_train, text="Noduri KAN").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm_train, textvariable=self.knots, width=16).grid(row=2, column=1, sticky="w")

        ttk.Label(frm_train, text="Putere spline").grid(row=2, column=2, sticky="w")
        ttk.Entry(frm_train, textvariable=self.spline_power, width=16).grid(row=2, column=3, sticky="w")

        ttk.Label(frm_train, text="Tip funcție").grid(row=2, column=4, sticky="w")
        ttk.Combobox(frm_train, textvariable=self.loss_type, values=["gauss_nll_var", "student_t_nll_var", "mse"], state="readonly", width=16).grid(row=2, column=5, sticky="w")

        ttk.Checkbutton(frm_train, text="Folosește logaritmul var.", variable=self.target_is_logvar).grid(row=2, column=6, sticky="w")
        ttk.Checkbutton(frm_train, text="Salvează GARCH", variable=self.save_garch).grid(row=2, column=7, sticky="w")

        ttk.Button(frm_train, text="Antrenează", command=self._start_training).grid(row=3, column=7, sticky="e")

        frm_pred = ttk.LabelFrame(self.root, text="Predicție")
        frm_pred.grid(row=2, column=0, sticky="nsew", **pad)
        for c in range(4):
            frm_pred.columnconfigure(c, weight=1)
        ttk.Button(frm_pred, text="Rulează predicție H=1", command=self._start_predict).grid(row=0, column=3, sticky="e")

        frm_out = ttk.LabelFrame(self.root, text="Consolă")
        frm_out.grid(row=3, column=0, sticky="nsew", **pad)
        frm_out.columnconfigure(0, weight=1)
        frm_out.rowconfigure(0, weight=1)
        self.txt = tk.Text(frm_out, height=22)
        self.txt.grid(row=0, column=0, sticky="nsew")
        try:
            self.txt.configure(font=("Menlo", 14))
        except Exception:
            self.txt.configure(font=("TkFixedFont", 14))
        sb = ttk.Scrollbar(frm_out, orient="vertical", command=self.txt.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=sb.set)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        self._log(f"Modele: {self.models_dir}")
        self._log(f"Ieșiri: {self.outputs_dir}")

        frm_viz = ttk.LabelFrame(self.root, text="Vizualizare")
        frm_viz.grid(row=4, column=0, sticky="ew", **pad)
        ttk.Button(frm_viz, text="Deschide graficele de antrenare", command=self._open_viz).pack(side="right")

    def _pick_csv(self):
        p = filedialog.askopenfilename(title="Selectează CSV", initialdir=self.outputs_dir, filetypes=[("CSV", "*.csv"), ("Toate fișierele", "*.*")])
        if p:
            self.csv_path.set(p)

    def _pick_model(self):
        p = filedialog.askopenfilename(title="Selectează model", initialdir=self.models_dir, filetypes=[("PKL", "*.pkl"), ("Toate", "*.*")])
        if p:
            self.model_path.set(p)

    def _start_training(self):
        try:
            csv = self.csv_path.get().strip()
            if not csv or not os.path.exists(csv):
                raise FileNotFoundError("Selectează fișierul CSV")
            arch = self.arch.get().strip()
            w = int(self.window.get())
            h = int(self.horizon.get())
            out_prefix = f"{arch}_w{w}_h{h}"
            run_dir = os.path.join(self.outputs_dir, out_prefix)
            os.makedirs(run_dir, exist_ok=True)
            garch_csv = os.path.join(run_dir, f"{out_prefix}_garch.csv") if self.save_garch.get() else None

            out = self.cmd.train(
                csv_path=csv,
                arch=arch,
                window=w,
                horizon=h,
                test_frac=0.2,
                out_prefix=out_prefix,
                epochs=int(self.epochs.get()),
                batch_size=int(self.batch_size.get()),
                lr=float(self.lr.get()),
                hidden_dim=128,
                hidden_layers=3,
                dropout=0.0,
                knots=int(self.knots.get()),
                spline_power=int(self.spline_power.get()),
                loss_type=self.loss_type.get(),
                target_is_logvar=bool(self.target_is_logvar.get()),
                patience=10,
                min_epochs=10,
                min_delta=1e-4,
                warmup_aux_epochs=10,
                joint_epochs=10,
                garch_save_path=garch_csv,
                verbose=True
            )

            self._log("Antrenarea s-a încheiat")
            self._log(f"Model: {out.get('model_path','')}")
            if "metrics" in out:
                self._log("Metrice:")
                for k, v in out["metrics"].items():
                    try:
                        self._log(f"  {k}: {float(v):.6f}")
                    except Exception:
                        self._log(f"  {k}: {v}")

            metrics_paths = out.get("metrics_path")
            curves_path = out.get("curves_path")

            paths = []
            if isinstance(curves_path, str) and curves_path:
                paths.append(curves_path)
            if isinstance(metrics_paths, list):
                paths.extend([p for p in metrics_paths if isinstance(p, str) and p])
            elif isinstance(metrics_paths, str) and metrics_paths:
                paths.append(metrics_paths)

            self.last_plots = paths
            self._log(f"Plot metrici: {metrics_paths}")
            self._log(f"Plot predicții: {curves_path}")
            self.model_path.set(out.get("model_path", ""))

        except Exception as e:
            messagebox.showerror("Eroare la antrenare", str(e))

    def _start_predict(self):
        try:
            csv = self.csv_path.get().strip()
            if not csv or not os.path.exists(csv):
                raise FileNotFoundError("Selectează fișierul CSV")
            model = self.model_path.get().strip()
            if not model or not os.path.exists(model):
                raise FileNotFoundError("Selectează un model salvat")
            out = self.cmd.predict(
                csv_path=csv,
                model_path=model,
                out_prefix="predict",
                outputs_dir=self.outputs_dir
            )
            self._log("Predicția s-a încheiat")
            self._log(f"CSV predicții H=1: {out.get('pred_csv','')}")
        except ValueError as ve:
            messagebox.showerror("Eroare parametri", str(ve))
        except Exception as e:
            messagebox.showerror("Eroare la predicție", str(e))

    def _open_viz(self):
        paths = []
        for p in (self.last_plots or []):
            if isinstance(p, str) and os.path.exists(p):
                paths.append(p)
        if not paths:
            pngs = [os.path.join(self.outputs_dir, f) for f in os.listdir(self.outputs_dir) if f.lower().endswith(".png")]
            pngs.sort(key=lambda p: os.path.getmtime(p))
            paths = pngs[-6:]
        if not paths:
            messagebox.showinfo("Vizualizare", "Nu s-au găsit grafice")
            return
        ImageCarousel(self.root, paths)

    def _log(self, msg: str):
        self.root.after(0, lambda: (self.txt.insert("end", str(msg) + "\n"), self.txt.see("end")))

if __name__ == "__main__":
    app = KANUI()
    app.root.mainloop()

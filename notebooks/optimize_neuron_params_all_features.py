#!/usr/bin/env python3
"""
Small neuron-parameter sensitivity search for the all-feature snnTorch EDA model.

Purpose
-------
Use the current best all-feature hyperparameters and test whether selected LIF
neuron settings have a measurable effect on LOSO performance.

Fixed best hyperparameters
--------------------------
hidden       = 64
beta         = 0.95
epochs       = 100
lr           = 0.005
encoding     = Gaussian population encoding
T            = 30
max_spikes   = 10
n_pop        = 3
sigma        = 0.25

Small neuron-parameter grid by default
--------------------------------------
threshold        = 0.75, 1.0, 1.25
reset_mechanism  = subtract, zero
surrogate_grad   = fast_sigmoid, atan

Total default configs = 3 × 2 × 2 = 12 configs.
Each config is evaluated with leave-one-subject-out cross-validation.

Run example
-----------
python notebooks/optimize_neuron_params_all_features.py \
  --input test/eda_features_29_samples.csv \
  --output neuron_param_search_all_features \
  --task_order "rest0,task1,task2,task3" \
  --device cpu

Notes
-----
By default, this script keeps the second Gaussian response normalization used in
your best model, because you verified that it improved prediction.
Use --response_normalization raw only for an explicit encoding ablation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
import zipfile
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate
except Exception as exc:
    raise ImportError("Install required packages with: pip install torch snntorch. Original error: " + repr(exc))


# ---------------------------------------------------------------------
# Reproducibility and device selection
# ---------------------------------------------------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print("MPS requested but unavailable. Falling back to CPU.")
        return "cpu"
    return "cpu"


# ---------------------------------------------------------------------
# Data loading and normalization
# ---------------------------------------------------------------------

def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if p.suffix.lower() == ".xlsx":
        return pd.read_excel(p, engine="openpyxl")
    if p.suffix.lower() == ".xls":
        return pd.read_excel(p, engine="xlrd")
    return pd.read_csv(p)


def infer_subject_column(df: pd.DataFrame) -> str:
    for col in ["subject_id", "Subject", "Sample", "subject", "sample"]:
        if col in df.columns:
            return col
    raise ValueError("No subject column found. Expected Sample, Subject, or subject_id.")


def infer_task_order(tasks: List[str], task_order: str | None) -> List[str]:
    if task_order:
        order = [x.strip() for x in task_order.split(",") if x.strip()]
        missing = sorted(set(order) - set(tasks))
        if missing:
            raise ValueError(f"Task labels not found in file: {missing}")
        return order
    if all(x in tasks for x in ["rest0", "task1", "task2", "task3"]):
        return ["rest0", "task1", "task2", "task3"]
    if all(x in tasks for x in ["Rest periods", "t1", "t2", "t3"]):
        return ["Rest periods", "t1", "t2", "t3"]
    return sorted(tasks)


def all_raw_features() -> List[str]:
    return [
        "Mean Tonic",
        "Max Slope",
        "Num Peaks",
        "Mean Amplitude",
        "Std Amplitude",
        "Total Power",
        "EDASymp Power",
    ]


def prepare_all_feature_data(args):
    df = read_table(args.input)
    if "Task" not in df.columns:
        raise ValueError("Input file must contain a Task column.")

    subject_col = infer_subject_column(df)
    df["subject_id"] = df[subject_col].astype(str)
    df["Task"] = df["Task"].astype(str)

    task_order = infer_task_order(list(pd.unique(df["Task"])), args.task_order)
    df = df[df["Task"].isin(task_order)].copy()
    df["Task"] = pd.Categorical(df["Task"], categories=task_order, ordered=True)
    df = df.sort_values(["subject_id", "Task"]).reset_index(drop=True)
    df["Task"] = df["Task"].astype(str)

    raw_cols = all_raw_features()
    missing = [c for c in raw_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required all-feature columns: {missing}")

    norm = df.copy()
    model_features = []
    for col in raw_cols:
        norm_col = col + "_norm"
        if args.use_existing_norm and norm_col in norm.columns:
            pass
        else:
            mn = norm.groupby("subject_id")[col].transform("min")
            mx = norm.groupby("subject_id")[col].transform("max")
            norm[norm_col] = ((norm[col] - mn) / (mx - mn + 1e-8)).clip(0, 1)
        model_features.append(norm_col)

    X = norm[model_features].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Model features contain NaN or inf.")
    if X.min() < -1e-8 or X.max() > 1 + 1e-8:
        raise ValueError("Model features must be normalized to [0, 1].")

    label_to_int = {lab: i for i, lab in enumerate(task_order)}
    y = norm["Task"].map(label_to_int).to_numpy(np.int64)
    groups = norm["subject_id"].to_numpy()
    return norm, X, y, groups, task_order, raw_cols, model_features


# ---------------------------------------------------------------------
# Spike encoding
# ---------------------------------------------------------------------

def deterministic_rate_encode(X: np.ndarray, T: int, max_spikes: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    S = np.zeros((X.shape[0], T, X.shape[1]), dtype=np.float32)
    for i in range(X.shape[0]):
        for f in range(X.shape[1]):
            n = int(np.round(float(X[i, f]) * max_spikes))
            if n > 0:
                S[i, np.linspace(0, T - 1, n).astype(int), f] = 1.0
    return S


def gaussian_population_encode(
    X: np.ndarray,
    T: int,
    max_spikes: int,
    n_pop: int = 3,
    sigma: float = 0.25,
    response_normalization: str = "max",
) -> np.ndarray:
    centers = np.linspace(0, 1, n_pop)
    responses = []
    for f in range(X.shape[1]):
        vals = X[:, f:f + 1]
        r = np.exp(-0.5 * ((vals - centers[None, :]) / sigma) ** 2)
        if response_normalization == "max":
            r = r / (r.max(axis=1, keepdims=True) + 1e-8)
        elif response_normalization == "raw":
            pass
        else:
            raise ValueError("response_normalization must be 'max' or 'raw'.")
        responses.append(r)
    X_pop = np.concatenate(responses, axis=1).astype(np.float32)
    return deterministic_rate_encode(X_pop, T=T, max_spikes=max_spikes)


# ---------------------------------------------------------------------
# SNN model with configurable neuron parameters
# ---------------------------------------------------------------------

def get_surrogate(name: str):
    name = name.lower()
    if name == "fast_sigmoid":
        return surrogate.fast_sigmoid()
    if name == "atan":
        return surrogate.atan()
    raise ValueError("surrogate_grad must be one of: fast_sigmoid, atan")


class SpikeSNN(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_classes: int,
        hidden: int,
        beta: float,
        threshold: float,
        reset_mechanism: str,
        surrogate_grad_name: str,
        learn_beta: bool = False,
        learn_threshold: bool = False,
    ):
        super().__init__()
        spike_grad = get_surrogate(surrogate_grad_name)
        self.fc1 = nn.Linear(n_inputs, hidden)
        self.lif1 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )
        self.fc2 = nn.Linear(hidden, n_classes)
        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            reset_mechanism=reset_mechanism,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, T, _ = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_counts = torch.zeros(batch, self.fc2.out_features, device=x.device)
        for t in range(T):
            cur1 = self.fc1(x[:, t, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_counts += spk2
        return out_counts


def train_model(S_train, y_train, n_classes, cfg, device, seed):
    set_all_seeds(seed)
    X_t = torch.tensor(S_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    model = SpikeSNN(
        n_inputs=S_train.shape[2],
        n_classes=n_classes,
        hidden=int(cfg["hidden"]),
        beta=float(cfg["beta"]),
        threshold=float(cfg["threshold"]),
        reset_mechanism=str(cfg["reset_mechanism"]),
        surrogate_grad_name=str(cfg["surrogate_grad"]),
        learn_beta=bool(cfg.get("learn_beta", False)),
        learn_threshold=bool(cfg.get("learn_threshold", False)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    crit = nn.CrossEntropyLoss()
    losses = []
    for _ in range(int(cfg["epochs"])):
        model.train()
        opt.zero_grad()
        logits = model(X_t)
        loss = crit(logits, y_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return model, losses


def predict_model(model, S_test, device):
    X_t = torch.tensor(S_test, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        out = model(X_t)
    return out.argmax(dim=1).cpu().numpy(), out.cpu().numpy()


def run_loso_config(X, y, groups, labels, cfg, device, seed, response_normalization):
    S_all = gaussian_population_encode(
        X,
        T=int(cfg["T"]),
        max_spikes=int(cfg["max_spikes"]),
        n_pop=int(cfg["n_pop"]),
        sigma=float(cfg["sigma"]),
        response_normalization=response_normalization,
    )
    yt, yp, rows, losses_all = [], [], [], []
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups), start=1):
        model, losses = train_model(S_all[tr], y[tr], len(labels), cfg, device, seed + fold)
        pred, out = predict_model(model, S_all[te], device)
        losses_all.append(losses)
        for j, idx in enumerate(te):
            row = {
                "fold": int(fold),
                "subject_id": str(groups[idx]),
                "y_true": int(y[idx]),
                "y_pred": int(pred[j]),
                "true_class": labels[int(y[idx])],
                "pred_class": labels[int(pred[j])],
            }
            for k, lab in enumerate(labels):
                row[f"out_spikes_{lab}"] = float(out[j, k])
            rows.append(row)
        yt.extend(y[te].tolist())
        yp.extend(pred.tolist())

    cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
    metrics = {
        **cfg,
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "weighted_f1": float(f1_score(yt, yp, average="weighted")),
        "n_predictions": int(len(yt)),
        "n_subjects": int(len(np.unique(groups))),
    }
    return metrics, pd.DataFrame(rows), cm, np.asarray(losses_all, dtype=float)


# ---------------------------------------------------------------------
# Grid and plotting
# ---------------------------------------------------------------------

def build_neuron_grid(args):
    configs = []
    base = {
        "hidden": int(args.hidden),
        "beta": float(args.beta),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "encoding": "population",
        "T": int(args.T),
        "max_spikes": int(args.max_spikes),
        "n_pop": int(args.n_pop),
        "sigma": float(args.sigma),
        "weight_decay": float(args.weight_decay),
        "learn_beta": False,
        "learn_threshold": False,
    }

    for threshold in args.threshold_values:
        for reset_mechanism in args.reset_mechanisms:
            for surrogate_grad in args.surrogate_grads:
                cfg = {
                    **base,
                    "threshold": float(threshold),
                    "reset_mechanism": str(reset_mechanism),
                    "surrogate_grad": str(surrogate_grad),
                }
                configs.append(cfg)

    if args.include_learnable_params:
        # Add only two extra cases to keep runtime modest.
        configs.append({
            **base,
            "threshold": 1.0,
            "reset_mechanism": "subtract",
            "surrogate_grad": "fast_sigmoid",
            "learn_beta": True,
            "learn_threshold": False,
        })
        configs.append({
            **base,
            "threshold": 1.0,
            "reset_mechanism": "subtract",
            "surrogate_grad": "fast_sigmoid",
            "learn_beta": False,
            "learn_threshold": True,
        })

    if args.max_configs is not None:
        configs = configs[: int(args.max_configs)]
    return configs


def save_confusion_matrix(cm, labels, outpath, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_results(results_df, outdir):
    top = results_df.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).head(20).copy()
    labels = []
    for _, r in top.iterrows():
        lb = f"thr={r.threshold}, reset={r.reset_mechanism}, sg={r.surrogate_grad}"
        if bool(r.get("learn_beta", False)):
            lb += ", learn_beta"
        if bool(r.get("learn_threshold", False)):
            lb += ", learn_thr"
        labels.append(lb)
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(y, top["balanced_accuracy"].values, color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Neuron-parameter sensitivity, all features")
    fig.tight_layout()
    fig.savefig(outdir / "top_neuron_parameter_configs.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.stripplot(data=results_df, x="threshold", y="balanced_accuracy", hue="reset_mechanism", dodge=True, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Effect of threshold and reset mechanism")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Balanced accuracy")
    ax.legend(title="Reset", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "threshold_reset_effect_stripplot.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="neuron_param_search_all_features")
    parser.add_argument("--task_order", default=None)
    parser.add_argument("--use_existing_norm", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_configs", type=int, default=None)

    # Fixed best all-feature hyperparameters.
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--max_spikes", type=int, default=10)
    parser.add_argument("--n_pop", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--response_normalization", choices=["max", "raw"], default="max")

    # Small neuron-parameter grid.
    parser.add_argument("--threshold_values", type=float, nargs="+", default=[0.75, 1.0, 1.25])
    parser.add_argument("--reset_mechanisms", type=str, nargs="+", default=["subtract", "zero"], choices=["subtract", "zero", "none"])
    parser.add_argument("--surrogate_grads", type=str, nargs="+", default=["fast_sigmoid", "atan"], choices=["fast_sigmoid", "atan"])
    parser.add_argument("--include_learnable_params", action="store_true", help="Add two extra configs: learn_beta=True and learn_threshold=True.")

    args = parser.parse_args()
    args.device = choose_device(args.device)
    print(f"Using device: {args.device}")

    out = Path(args.output)
    out.mkdir(exist_ok=True)
    pred_dir = out / "predictions"
    cm_dir = out / "confusion_matrices"
    pred_dir.mkdir(exist_ok=True)
    cm_dir.mkdir(exist_ok=True)

    norm, X, y, groups, labels, raw_features, model_features = prepare_all_feature_data(args)
    norm.to_csv(out / "features_used_all_feature_neuron_param_search.csv", index=False)

    configs = build_neuron_grid(args)
    print(f"Rows={len(norm)}, subjects={len(np.unique(groups))}, classes={labels}")
    print(f"All features ({len(model_features)}): {model_features}")
    print(f"Input channels after population encoding: {X.shape[1] * args.n_pop}")
    print(f"Running {len(configs)} neuron-parameter configs")

    t0 = time.time()
    rows = []
    best = None
    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] threshold={cfg['threshold']}, reset={cfg['reset_mechanism']}, surrogate={cfg['surrogate_grad']}, learn_beta={cfg['learn_beta']}, learn_threshold={cfg['learn_threshold']}", flush=True)
        metrics, preds, cm, losses = run_loso_config(
            X, y, groups, labels, cfg, args.device, args.seed,
            response_normalization=args.response_normalization,
        )
        row = {
            **metrics,
            "config_id": int(i),
            "feature_group": "all",
            "n_features": int(X.shape[1]),
            "input_channels": int(X.shape[1] * args.n_pop),
        }
        rows.append(row)
        if best is None or (row["balanced_accuracy"], row["macro_f1"]) > (best[0]["balanced_accuracy"], best[0]["macro_f1"]):
            best = (row, preds, cm, losses, cfg)
        pd.DataFrame(rows).sort_values(["balanced_accuracy", "macro_f1"], ascending=False).to_csv(out / "neuron_param_search_results_partial.csv", index=False)

    results_df = pd.DataFrame(rows).sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    best_row, best_preds, best_cm, best_losses, best_cfg = best

    results_df.to_csv(out / "neuron_param_search_results.csv", index=False)
    best_preds.to_csv(pred_dir / "best_neuron_param_predictions.csv", index=False)
    pd.DataFrame(best_cm, index=labels, columns=labels).to_csv(cm_dir / "best_neuron_param_confusion_matrix.csv")
    pd.DataFrame(best_losses).to_csv(out / "best_neuron_param_losses.csv", index=False)
    save_confusion_matrix(best_cm, labels, cm_dir / "best_neuron_param_confusion_matrix.png", "Best neuron-parameter config")
    plot_results(results_df, out)

    summary = {
        "input_file": args.input,
        "device_used": args.device,
        "n_rows": int(len(norm)),
        "n_subjects": int(len(np.unique(groups))),
        "classes": labels,
        "raw_feature_cols": raw_features,
        "model_feature_cols": model_features,
        "fixed_best_hyperparameters": {
            "hidden": args.hidden,
            "beta": args.beta,
            "epochs": args.epochs,
            "lr": args.lr,
            "T": args.T,
            "max_spikes": args.max_spikes,
            "n_pop": args.n_pop,
            "sigma": args.sigma,
            "response_normalization": args.response_normalization,
        },
        "searched_neuron_parameters": {
            "threshold_values": args.threshold_values,
            "reset_mechanisms": args.reset_mechanisms,
            "surrogate_grads": args.surrogate_grads,
            "include_learnable_params": bool(args.include_learnable_params),
        },
        "n_configs_run": int(len(configs)),
        "elapsed_seconds": float(time.time() - t0),
        "best_config": best_row,
        "note": "This is a small sensitivity search over selected neuron parameters while keeping the current best all-feature hyperparameters fixed.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    zip_path = Path(str(out) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            z.write(p, arcname=f"{out.name}/{p.relative_to(out)}")

    print("\nTop neuron-parameter configurations:")
    print(results_df.head(20).to_string(index=False))
    print("\nBest configuration:")
    print(json.dumps(best_row, indent=2, default=str))
    print(f"\nOutputs saved to: {out}")
    print(f"Zip saved to: {zip_path}")


if __name__ == "__main__":
    main()

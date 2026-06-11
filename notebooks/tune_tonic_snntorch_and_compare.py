#!/usr/bin/env python3
"""
Optimize a tonic-only snnTorch SNN for EDA task classification.

Tonic feature group
-------------------
Mean Tonic
Max Slope

Default tuning grid
-------------------
hidden neurons: 8, 16, 32, 64
beta:           0.85, 0.90, 0.95
epochs:         50, 100, 200
learning rate:  0.0005, 0.001, 0.005
encoding:       Gaussian population encoding only
max_spikes:     4, 6, 10
T:              30
n_pop:          3
sigma:          0.25

Evaluation
----------
Leave-one-subject-out cross-validation using Sample, Subject, or subject_id.
No data augmentation.

The script also evaluates classical ML baselines and a spike-count delta readout
on the same tonic features. The spike-delta readout uses the best SNN encoding
settings found during tonic-only tuning.

Important
---------
By default, this script uses max-normalized Gaussian population responses,
because this matches the encoding used in your previous best SNN run.
Use --response_normalization raw for an ablation with raw Gaussian responses.

Example
-------
python tune_tonic_snntorch_and_compare.py \
  --input test/eda_features_29_samples.csv \
  --output tonic_optimization_outputs \
  --task_order "rest0,task1,task2,task3" \
  --device cpu
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
import zipfile
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC

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
# Data loading
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
    if all(x in tasks for x in ["Rest periods", "t1", "t2", "t3", "t4", "t5"]):
        return ["Rest periods", "t1", "t2", "t3", "t4", "t5"]
    return sorted(tasks)


def prepare_tonic_data(args):
    df = read_table(args.input)
    if "Task" not in df.columns:
        raise ValueError("Input file must contain a Task column.")

    subject_col = infer_subject_column(df)
    df["subject_id"] = df[subject_col].astype(str)
    df["Task"] = df["Task"].astype(str)

    label_order = infer_task_order(list(pd.unique(df["Task"])), args.task_order)
    df = df[df["Task"].isin(label_order)].copy()
    df["Task"] = pd.Categorical(df["Task"], categories=label_order, ordered=True)
    df = df.sort_values(["subject_id", "Task"]).reset_index(drop=True)
    df["Task"] = df["Task"].astype(str)

    tonic_raw = ["Mean Tonic", "Max Slope"]
    missing = [c for c in tonic_raw if c not in df.columns]
    if missing:
        raise ValueError(f"Missing tonic feature columns: {missing}")

    norm = df.copy()
    tonic_norm = []
    for col in tonic_raw:
        norm_col = col + "_norm"
        if args.use_existing_norm and norm_col in norm.columns:
            pass
        else:
            mn = norm.groupby("subject_id")[col].transform("min")
            mx = norm.groupby("subject_id")[col].transform("max")
            norm[norm_col] = ((norm[col] - mn) / (mx - mn + 1e-8)).clip(0, 1)
        tonic_norm.append(norm_col)

    X = norm[tonic_norm].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Tonic feature matrix contains NaN or inf.")
    if X.min() < -1e-8 or X.max() > 1 + 1e-8:
        raise ValueError("Tonic features must be normalized to [0, 1].")

    label_to_int = {lab: i for i, lab in enumerate(label_order)}
    y = norm["Task"].map(label_to_int).to_numpy(np.int64)
    groups = norm["subject_id"].to_numpy()
    return norm, X, y, groups, label_order, tonic_raw, tonic_norm


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
# snnTorch model
# ---------------------------------------------------------------------

class SpikeSNN(nn.Module):
    def __init__(self, n_inputs: int, n_classes: int, hidden: int, beta: float):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(n_inputs, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden, n_classes)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

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


def train_snn(S_train, y_train, n_classes, cfg, device, seed):
    set_all_seeds(seed)
    X_t = torch.tensor(S_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    model = SpikeSNN(S_train.shape[2], n_classes, int(cfg["hidden"]), float(cfg["beta"])).to(device)
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


def predict_snn(model, S_test, device):
    X_t = torch.tensor(S_test, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        out = model(X_t)
    return out.argmax(dim=1).cpu().numpy(), out.cpu().numpy()


def run_snn_loso(X, y, groups, labels, cfg, device, response_normalization, seed):
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
        if set(groups[tr]).intersection(set(groups[te])):
            raise RuntimeError("Subject leakage detected.")
        model, losses = train_snn(S_all[tr], y[tr], len(labels), cfg, device, seed + fold)
        pred, out = predict_snn(model, S_all[te], device)
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
# Spike-delta readout
# ---------------------------------------------------------------------

def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def train_delta_readout(S_train, y_train, n_classes, epochs=400, lr=0.05, l2=1e-4, seed=42):
    rng = np.random.default_rng(seed)
    X_counts = S_train.sum(axis=1).astype(float)
    scale = X_counts.max(axis=0, keepdims=True) + 1e-8
    X_counts_scaled = X_counts / scale
    W = rng.normal(0, 0.05, size=(X_counts_scaled.shape[1], n_classes))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y_train]
    losses = []
    for _ in range(int(epochs)):
        logits = X_counts_scaled @ W + b
        P = softmax_np(logits)
        loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1)) + l2 * np.sum(W * W)
        grad_logits = (P - Y) / len(y_train)
        grad_W = X_counts_scaled.T @ grad_logits + 2 * l2 * W
        grad_b = grad_logits.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b
        losses.append(float(loss))
    return W, b, scale, losses


def predict_delta_readout(S_test, W, b, scale):
    X_counts = S_test.sum(axis=1).astype(float)
    X_counts_scaled = X_counts / scale
    logits = X_counts_scaled @ W + b
    return logits.argmax(axis=1), logits


def run_delta_loso(X, y, groups, labels, cfg, response_normalization, seed, delta_epochs, delta_lr, delta_l2):
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
        W, b, scale, losses = train_delta_readout(
            S_all[tr], y[tr], len(labels),
            epochs=delta_epochs, lr=delta_lr, l2=delta_l2, seed=seed + fold,
        )
        pred, logits = predict_delta_readout(S_all[te], W, b, scale)
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
                row[f"logit_{lab}"] = float(logits[j, k])
            rows.append(row)
        yt.extend(y[te].tolist())
        yp.extend(pred.tolist())
    cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
    metrics = {
        "method": "spike_delta",
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "weighted_f1": float(f1_score(yt, yp, average="weighted")),
        "n_predictions": int(len(yt)),
    }
    return metrics, pd.DataFrame(rows), cm, np.asarray(losses_all, dtype=float)


# ---------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------

def run_classical_loso(X, y, groups, labels):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVM": SVC(kernel="linear", class_weight="balanced"),
        "RBFSVM": SVC(kernel="rbf", class_weight="balanced"),
    }
    out = []
    for name, clf in models.items():
        yt, yp, rows = [], [], []
        for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups), start=1):
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            for j, idx in enumerate(te):
                rows.append({
                    "fold": int(fold),
                    "subject_id": str(groups[idx]),
                    "y_true": int(y[idx]),
                    "y_pred": int(pred[j]),
                    "true_class": labels[int(y[idx])],
                    "pred_class": labels[int(pred[j])],
                })
            yt.extend(y[te].tolist())
            yp.extend(pred.tolist())
        cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
        metrics = {
            "method": name,
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "weighted_f1": float(f1_score(yt, yp, average="weighted")),
            "n_predictions": int(len(yt)),
        }
        out.append((metrics, pd.DataFrame(rows), cm))
    return out


# ---------------------------------------------------------------------
# Grid, plotting, saving
# ---------------------------------------------------------------------

def build_grid(args):
    configs = []
    for hidden in args.hidden_values:
        for beta in args.beta_values:
            for epochs in args.epoch_values:
                for lr in args.lr_values:
                    for max_spikes in args.max_spikes_values:
                        configs.append({
                            "hidden": int(hidden),
                            "beta": float(beta),
                            "epochs": int(epochs),
                            "lr": float(lr),
                            "encoding": "population",
                            "T": int(args.T),
                            "max_spikes": int(max_spikes),
                            "n_pop": int(args.n_pop),
                            "sigma": float(args.sigma),
                            "weight_decay": float(args.weight_decay),
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


def plot_top_snn_configs(results_df, outpath):
    top = results_df.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).head(20)
    labels = [f"h={int(r.hidden)}, beta={r.beta}, ep={int(r.epochs)}, lr={r.lr}, sp={int(r.max_spikes)}" for _, r in top.iterrows()]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y, top["balanced_accuracy"].values, color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Top tonic-only snnTorch configurations")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_model_comparison(metrics_df, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = metrics_df.sort_values("balanced_accuracy", ascending=False)
    sns.barplot(data=plot_df, x="balanced_accuracy", y="method", ax=ax, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy")
    ax.set_ylabel("Model")
    ax.set_title("Tonic-only model comparison")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="tonic_snntorch_optimization_outputs")
    parser.add_argument("--task_order", default=None)
    parser.add_argument("--use_existing_norm", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_configs", type=int, default=None)

    parser.add_argument("--hidden_values", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--beta_values", type=float, nargs="+", default=[0.85, 0.90, 0.95])
    parser.add_argument("--epoch_values", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--lr_values", type=float, nargs="+", default=[0.0005, 0.001, 0.005])
    parser.add_argument("--max_spikes_values", type=int, nargs="+", default=[4, 6, 10])
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n_pop", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--response_normalization", choices=["max", "raw"], default="max")

    parser.add_argument("--delta_epochs", type=int, default=400)
    parser.add_argument("--delta_lr", type=float, default=0.05)
    parser.add_argument("--delta_l2", type=float, default=1e-4)

    args = parser.parse_args()
    args.device = choose_device(args.device)
    print(f"Using device: {args.device}")

    out = Path(args.output)
    out.mkdir(exist_ok=True)
    cm_dir = out / "confusion_matrices"
    pred_dir = out / "predictions"
    cm_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)

    norm, X, y, groups, labels, tonic_raw, tonic_norm = prepare_tonic_data(args)
    norm.to_csv(out / "tonic_features_normalized.csv", index=False)

    configs = build_grid(args)
    print(f"Rows={len(norm)}, subjects={len(np.unique(groups))}, classes={labels}")
    print(f"Tonic features: {tonic_norm}")
    print(f"Input channels after population encoding: {X.shape[1] * args.n_pop}")
    print(f"Running {len(configs)} tonic-only snnTorch configurations.")

    t0 = time.time()
    snn_rows = []
    best = None
    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {cfg}", flush=True)
        metrics, preds, cm, losses = run_snn_loso(
            X, y, groups, labels, cfg, args.device,
            response_normalization=args.response_normalization,
            seed=args.seed,
        )
        row = {
            **metrics,
            "method": "snnTorch",
            "config_id": int(i),
            "feature_group": "tonic",
            "n_features": int(X.shape[1]),
            "input_channels": int(X.shape[1] * args.n_pop),
        }
        snn_rows.append(row)
        if best is None or (row["balanced_accuracy"], row["macro_f1"]) > (best[0]["balanced_accuracy"], best[0]["macro_f1"]):
            best = (row, preds, cm, losses, cfg)
        pd.DataFrame(snn_rows).sort_values(["balanced_accuracy", "macro_f1"], ascending=False).to_csv(out / "tonic_snntorch_tuning_results_partial.csv", index=False)

    snn_results_df = pd.DataFrame(snn_rows).sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    best_row, best_preds, best_cm, best_losses, best_cfg = best

    snn_results_df.to_csv(out / "tonic_snntorch_tuning_results.csv", index=False)
    best_preds.to_csv(pred_dir / "predictions_tonic_best_snnTorch.csv", index=False)
    pd.DataFrame(best_cm, index=labels, columns=labels).to_csv(cm_dir / "confusion_matrix_tonic_best_snnTorch.csv")
    save_confusion_matrix(best_cm, labels, cm_dir / "confusion_matrix_tonic_best_snnTorch.png", "Best tonic-only snnTorch")
    pd.DataFrame(best_losses).to_csv(out / "losses_tonic_best_snnTorch.csv", index=False)

    # Compare best SNN configuration against spike-delta and classical baselines.
    delta_metrics, delta_preds, delta_cm, delta_losses = run_delta_loso(
        X, y, groups, labels, best_cfg,
        response_normalization=args.response_normalization,
        seed=args.seed,
        delta_epochs=args.delta_epochs,
        delta_lr=args.delta_lr,
        delta_l2=args.delta_l2,
    )
    delta_metrics = {**delta_metrics, "feature_group": "tonic", "n_features": int(X.shape[1]), "input_channels": int(X.shape[1] * args.n_pop)}
    delta_preds.to_csv(pred_dir / "predictions_tonic_spike_delta.csv", index=False)
    pd.DataFrame(delta_cm, index=labels, columns=labels).to_csv(cm_dir / "confusion_matrix_tonic_spike_delta.csv")
    save_confusion_matrix(delta_cm, labels, cm_dir / "confusion_matrix_tonic_spike_delta.png", "Tonic spike-delta")
    pd.DataFrame(delta_losses).to_csv(out / "losses_tonic_spike_delta.csv", index=False)

    classical_results = run_classical_loso(X, y, groups, labels)
    comparison_rows = [best_row, delta_metrics]
    for metrics, preds, cm in classical_results:
        metrics = {**metrics, "feature_group": "tonic", "n_features": int(X.shape[1]), "input_channels": int(X.shape[1] * args.n_pop)}
        comparison_rows.append(metrics)
        method = metrics["method"]
        preds.to_csv(pred_dir / f"predictions_tonic_{method}.csv", index=False)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_dir / f"confusion_matrix_tonic_{method}.csv")
        save_confusion_matrix(cm, labels, cm_dir / f"confusion_matrix_tonic_{method}.png", f"Tonic {method}")

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    comparison_df.to_csv(out / "tonic_best_snn_vs_delta_and_classical_metrics.csv", index=False)

    plot_top_snn_configs(snn_results_df, out / "top_tonic_snntorch_configurations.png")
    plot_model_comparison(comparison_df, out / "tonic_model_comparison_balanced_accuracy.png")

    summary = {
        "input_file": args.input,
        "device_used": args.device,
        "n_rows": int(len(norm)),
        "n_subjects": int(len(np.unique(groups))),
        "classes": labels,
        "tonic_raw_features": tonic_raw,
        "tonic_normalized_features": tonic_norm,
        "response_normalization": args.response_normalization,
        "n_snn_configs_run": int(len(configs)),
        "best_tonic_snntorch_config": best_row,
        "best_tonic_snntorch_hyperparameters": best_cfg,
        "spike_delta_config": {
            "epochs": args.delta_epochs,
            "lr": args.delta_lr,
            "l2": args.delta_l2,
            "encoding": "population",
            "T": best_cfg["T"],
            "max_spikes": best_cfg["max_spikes"],
            "n_pop": best_cfg["n_pop"],
            "sigma": best_cfg["sigma"],
            "response_normalization": args.response_normalization,
        },
        "comparison_metrics": comparison_df.to_dict(orient="records"),
        "elapsed_seconds": float(time.time() - t0),
        "note": "This script tunes snnTorch only on tonic features and then compares the best tonic SNN to spike-delta and classical ML on the same tonic features.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    zip_path = Path(str(out) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            z.write(p, arcname=f"{out.name}/{p.relative_to(out)}")

    print("\nBest tonic-only snnTorch configurations:")
    print(snn_results_df.head(15).to_string(index=False))
    print("\nBest tonic SNN versus spike-delta and classical models:")
    print(comparison_df.to_string(index=False))
    print(f"\nOutputs saved to: {out}")
    print(f"Zip saved to: {zip_path}")


if __name__ == "__main__":
    main()

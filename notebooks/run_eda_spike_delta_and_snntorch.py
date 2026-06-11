#!/usr/bin/env python3
"""
EDA spike-based classification with two neuromorphic-style models:

1. Spike-count delta-rule readout
   - Fast NumPy implementation
   - Spike encoding + softmax delta-rule readout

2. snnTorch SNN
   - Real snnTorch LIF neurons
   - Surrogate-gradient training with CrossEntropyLoss
   - Output spike-count readout

Both are evaluated with leave-one-subject-out cross-validation using Sample as subject_id.
No data augmentation is used.

Expected input CSV columns:
    Sample, Task, Mean Tonic, Max Slope, Num Peaks,
    Mean Amplitude, Std Amplitude, Total Power, EDASymp Power

Run:
    python run_eda_spike_delta_and_snntorch.py --input eda_features.csv --output eda_snn_outputs

Optional:
    python run_eda_spike_delta_and_snntorch.py --input eda_features.csv --output eda_snn_outputs --snn_full_grid
"""

from __future__ import annotations

import argparse
import json
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Torch/snnTorch are optional. If unavailable, the delta-rule branch still runs.
try:
    import torch
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate

    HAS_SNNTORCH = True
except Exception as exc:  # pragma: no cover
    HAS_SNNTORCH = False
    TORCH_IMPORT_ERROR = repr(exc)


# ---------------------------------------------------------------------
# Data loading and normalization
# ---------------------------------------------------------------------

def load_and_prepare_data(input_csv: str):
    raw = pd.read_csv(input_csv)

    required_cols = {
        "Sample",
        "Task",
        "Mean Tonic",
        "Max Slope",
        "Num Peaks",
        "Mean Amplitude",
        "Std Amplitude",
        "Total Power",
        "EDASymp Power",
    }
    missing = required_cols.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    raw["subject_id"] = raw["Sample"].astype(str)

    label_order = ["rest0", "task1", "task2", "task3"]
    raw = raw[raw["Task"].isin(label_order)].copy()
    raw["Task"] = pd.Categorical(raw["Task"], categories=label_order, ordered=True)
    raw = raw.sort_values(["subject_id", "Task"]).reset_index(drop=True)
    raw["Task"] = raw["Task"].astype(str)

    raw_feature_cols = [
        "Mean Tonic",
        "Max Slope",
        "Num Peaks",
        "Mean Amplitude",
        "Std Amplitude",
        "Total Power",
        "EDASymp Power",
    ]

    norm = raw.copy()
    for col in raw_feature_cols:
        mn = norm.groupby("subject_id")[col].transform("min")
        mx = norm.groupby("subject_id")[col].transform("max")
        norm[col + "_norm"] = ((norm[col] - mn) / (mx - mn + 1e-8)).clip(0, 1)

    # Optional derived normalized EDASymp ratio.
    norm["EDASymp_Ratio"] = norm["EDASymp Power"] / (norm["Total Power"] + 1e-12)
    mn = norm.groupby("subject_id")["EDASymp_Ratio"].transform("min")
    mx = norm.groupby("subject_id")["EDASymp_Ratio"].transform("max")
    norm["EDASymp_Ratio_norm"] = ((norm["EDASymp_Ratio"] - mn) / (mx - mn + 1e-8)).clip(0, 1)

    feature_sets = {
        "all_7_norm": [c + "_norm" for c in raw_feature_cols],
        "all_7_plus_symp_ratio": [c + "_norm" for c in raw_feature_cols] + ["EDASymp_Ratio_norm"],
        "no_count_plus_symp_ratio": [c + "_norm" for c in raw_feature_cols if c != "Num Peaks"]
        + ["EDASymp_Ratio_norm"],
    }

    label_to_int = {lab: i for i, lab in enumerate(label_order)}
    y = norm["Task"].map(label_to_int).to_numpy(np.int64)
    groups = norm["subject_id"].to_numpy()

    return norm, y, groups, label_order, raw_feature_cols, feature_sets


# ---------------------------------------------------------------------
# Spike encoding
# ---------------------------------------------------------------------

def deterministic_rate_encode(X: np.ndarray, T: int = 50, max_spikes: int = 15) -> np.ndarray:
    """Convert normalized features in [0, 1] to deterministic rate-coded spikes."""
    X = np.asarray(X, dtype=float)

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input feature matrix contains NaN or inf.")
    if X.min() < -1e-8 or X.max() > 1 + 1e-8:
        raise ValueError(f"Input features must be in [0, 1]. Got min={X.min()}, max={X.max()}.")

    spikes = np.zeros((X.shape[0], T, X.shape[1]), dtype=np.float32)
    for i in range(X.shape[0]):
        for f in range(X.shape[1]):
            n_spikes = int(np.round(X[i, f] * max_spikes))
            if n_spikes > 0:
                spike_times = np.linspace(0, T - 1, n_spikes).astype(int)
                spikes[i, spike_times, f] = 1.0
    return spikes


def gaussian_population_encode(
    X: np.ndarray,
    T: int = 50,
    max_spikes: int = 10,
    n_pop: int = 3,
    sigma: float = 0.25,
) -> np.ndarray:
    """Expand each scalar feature into Gaussian-tuned channels, then rate encode."""
    X = np.asarray(X, dtype=float)

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input feature matrix contains NaN or inf.")

    centers = np.linspace(0, 1, n_pop)
    responses = []
    for f in range(X.shape[1]):
        vals = X[:, f : f + 1]
        r = np.exp(-0.5 * ((vals - centers[None, :]) / sigma) ** 2)
        r = r / (r.max(axis=1, keepdims=True) + 1e-8)
        responses.append(r)

    X_pop = np.concatenate(responses, axis=1).astype(np.float32)
    return deterministic_rate_encode(X_pop, T=T, max_spikes=max_spikes)


def make_spikes(X: np.ndarray, encoding: str, T: int, max_spikes: int) -> np.ndarray:
    if encoding == "rate":
        return deterministic_rate_encode(X, T=T, max_spikes=max_spikes)
    if encoding == "population":
        return gaussian_population_encode(X, T=T, max_spikes=max_spikes)
    raise ValueError(f"Unknown encoding: {encoding}")


# ---------------------------------------------------------------------
# Delta-rule readout
# ---------------------------------------------------------------------

def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def train_delta_readout(
    S_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    epochs: int = 400,
    lr: float = 0.05,
    l2: float = 1e-4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Train a compact spike-count readout using a multiclass delta-rule update."""
    rng = np.random.default_rng(seed)
    X_counts = S_train.sum(axis=1).astype(float)
    X_counts = X_counts / (X_counts.max(axis=0, keepdims=True) + 1e-8)

    W = rng.normal(0, 0.05, size=(X_counts.shape[1], n_classes))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y_train]

    losses = []
    for _ in range(epochs):
        logits = X_counts @ W + b
        P = softmax_np(logits)
        loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1)) + l2 * np.sum(W * W)

        grad_logits = (P - Y) / len(y_train)
        grad_W = X_counts.T @ grad_logits + 2 * l2 * W
        grad_b = grad_logits.sum(axis=0)

        W -= lr * grad_W
        b -= lr * grad_b
        losses.append(float(loss))

    return W, b, losses


def predict_delta_readout(S: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_counts = S.sum(axis=1).astype(float)
    X_counts = X_counts / (X_counts.max(axis=0, keepdims=True) + 1e-8)
    logits = X_counts @ W + b
    y_pred = logits.argmax(axis=1)
    return y_pred, logits


def run_spike_delta_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_order: List[str],
    encoding: str,
    T: int,
    max_spikes: int,
    epochs: int,
    lr: float,
    seed: int = 100,
) -> Tuple[Dict, pd.DataFrame, np.ndarray, np.ndarray]:
    all_true, all_pred, rows, fold_losses = [], [], [], []
    logo = LeaveOneGroupOut()

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        if set(groups[train_idx]).intersection(set(groups[test_idx])):
            raise RuntimeError("Subject leakage detected.")

        S_train = make_spikes(X[train_idx], encoding=encoding, T=T, max_spikes=max_spikes)
        S_test = make_spikes(X[test_idx], encoding=encoding, T=T, max_spikes=max_spikes)

        W, b, losses = train_delta_readout(
            S_train=S_train,
            y_train=y[train_idx],
            n_classes=len(label_order),
            epochs=epochs,
            lr=lr,
            seed=seed + fold,
        )
        y_pred, logits = predict_delta_readout(S_test, W, b)
        fold_losses.append(losses)

        for i, idx in enumerate(test_idx):
            row = {
                "fold": fold,
                "subject_id": groups[idx],
                "y_true": int(y[idx]),
                "y_pred": int(y_pred[i]),
                "pred_Task": label_order[int(y_pred[i])],
            }
            for k, lab in enumerate(label_order):
                row[f"logit_{lab}"] = float(logits[i, k])
            rows.append(row)

        all_true.extend(y[test_idx].tolist())
        all_pred.extend(y_pred.tolist())

    metrics = {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro")),
        "weighted_f1": float(f1_score(all_true, all_pred, average="weighted")),
        "n_predictions": int(len(all_true)),
    }
    cm = confusion_matrix(all_true, all_pred, labels=np.arange(len(label_order)))
    return metrics, pd.DataFrame(rows), cm, np.asarray(fold_losses, dtype=float)


# ---------------------------------------------------------------------
# snnTorch SNN branch
# ---------------------------------------------------------------------

if HAS_SNNTORCH:

    class SpikeSNN(nn.Module):
        """Simple snnTorch SNN: input spikes -> hidden LIF -> output LIF."""

        def __init__(self, n_inputs: int, n_classes: int, hidden: int = 64, beta: float = 0.9):
            super().__init__()
            spike_grad = surrogate.fast_sigmoid()
            self.fc1 = nn.Linear(n_inputs, hidden)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc2 = nn.Linear(hidden, n_classes)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x shape: [batch, T, n_inputs]
            batch_size, T, _ = x.shape
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            out_spike_counts = torch.zeros(batch_size, self.fc2.out_features, device=x.device)

            for t in range(T):
                cur1 = self.fc1(x[:, t, :])
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                out_spike_counts += spk2

            return out_spike_counts


    def train_spiking_network(
        S_train: np.ndarray,
        y_train: np.ndarray,
        n_classes: int,
        epochs: int = 100,
        lr: float = 1e-3,
        beta: float = 0.9,
        hidden: int = 64,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_train = torch.tensor(S_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)

        model = SpikeSNN(n_inputs=S_train.shape[2], n_classes=n_classes, hidden=hidden, beta=beta)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        losses = []
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        return model, losses


    def predict_spiking_network(model, S_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_test = torch.tensor(S_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            y_pred = logits.argmax(dim=1).cpu().numpy()
            logits_np = logits.cpu().numpy()
        return y_pred, logits_np


def run_spike_snn_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_order: List[str],
    encoding: str,
    T: int,
    max_spikes: int,
    epochs: int,
    lr: float,
    hidden: int,
    beta: float,
    seed: int = 100,
) -> Tuple[Dict, pd.DataFrame, np.ndarray, np.ndarray]:
    if not HAS_SNNTORCH:
        raise ImportError(
            "snnTorch/PyTorch is not available. Install with: pip install torch snntorch. "
            f"Import error was: {TORCH_IMPORT_ERROR}"
        )

    all_true, all_pred, rows, fold_losses = [], [], [], []
    logo = LeaveOneGroupOut()

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        if set(groups[train_idx]).intersection(set(groups[test_idx])):
            raise RuntimeError("Subject leakage detected.")

        S_train = make_spikes(X[train_idx], encoding=encoding, T=T, max_spikes=max_spikes)
        S_test = make_spikes(X[test_idx], encoding=encoding, T=T, max_spikes=max_spikes)

        model, losses = train_spiking_network(
            S_train=S_train,
            y_train=y[train_idx],
            n_classes=len(label_order),
            epochs=epochs,
            lr=lr,
            beta=beta,
            hidden=hidden,
            seed=seed + fold,
        )
        y_pred, logits = predict_spiking_network(model, S_test)
        fold_losses.append(losses)

        for i, idx in enumerate(test_idx):
            row = {
                "fold": fold,
                "subject_id": groups[idx],
                "y_true": int(y[idx]),
                "y_pred": int(y_pred[i]),
                "pred_Task": label_order[int(y_pred[i])],
            }
            for k, lab in enumerate(label_order):
                row[f"logit_{lab}"] = float(logits[i, k])
            rows.append(row)

        all_true.extend(y[test_idx].tolist())
        all_pred.extend(y_pred.tolist())

    metrics = {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
        "macro_f1": float(f1_score(all_true, all_pred, average="macro")),
        "weighted_f1": float(f1_score(all_true, all_pred, average="weighted")),
        "n_predictions": int(len(all_true)),
    }
    cm = confusion_matrix(all_true, all_pred, labels=np.arange(len(label_order)))
    return metrics, pd.DataFrame(rows), cm, np.asarray(fold_losses, dtype=float)


# ---------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------

def run_classical_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVM": SVC(kernel="linear", class_weight="balanced"),
        "RBFSVM": SVC(kernel="rbf", class_weight="balanced"),
    }
    rows = []

    for name, clf in models.items():
        all_true, all_pred = [], []
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            all_true.extend(y[test_idx].tolist())
            all_pred.extend(pred.tolist())

        rows.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(all_true, all_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
                "macro_f1": float(f1_score(all_true, all_pred, average="macro")),
                "weighted_f1": float(f1_score(all_true, all_pred, average="weighted")),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_top_configs(results_df: pd.DataFrame, output_path: Path, title: str) -> None:
    top = results_df.sort_values("balanced_accuracy", ascending=False).head(12).copy()
    labels = [
        f"{r['method']} | {r['feature_set']} | {r['encoding']} | T={int(r['T'])} | sp={int(r['max_spikes'])}"
        for _, r in top.iterrows()
    ]
    vals = top["balanced_accuracy"].values
    ypos = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ypos, vals, color="steelblue")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 0.95), i, f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, label_order: List[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_order, yticklabels=label_order, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_heatmap(norm_df: pd.DataFrame, feature_cols: List[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(norm_df[feature_cols].to_numpy(), vmin=0, vmax=1, cmap="viridis", xticklabels=feature_cols, yticklabels=False, ax=ax)
    ax.set_title("Normalized features for selected feature set")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_example_spike_raster(X0: np.ndarray, encoding: str, T: int, max_spikes: int, output_path: Path, title: str) -> None:
    if encoding == "population":
        S0 = gaussian_population_encode(X0, T=T, max_spikes=max_spikes)[0]
    else:
        S0 = deterministic_rate_encode(X0, T=T, max_spikes=max_spikes)[0]

    time_idx, channel_idx = np.where(S0.T == 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(time_idx, channel_idx, marker="|", s=40)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Input channel")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_loss(losses: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.mean(losses, axis=0))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eda_features.csv")
    parser.add_argument("--output", default="eda_spike_delta_snntorch_outputs")
    parser.add_argument("--skip_snn", action="store_true", help="Skip snnTorch branch and run only delta-rule readout.")
    parser.add_argument("--snn_full_grid", action="store_true", help="Run snnTorch branch over the full delta grid. Slower.")
    parser.add_argument("--snn_epochs", type=int, default=100, help="Training epochs for snnTorch branch.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    norm, y, groups, label_order, raw_feature_cols, feature_sets = load_and_prepare_data(str(input_path))
    norm.to_csv(output_dir / "features_normalized_with_ratio.csv", index=False)

    # Full delta grid, similar to the earlier successful spike-delta pipeline.
    delta_configs = []
    for feature_set_name in feature_sets:
        for encoding in ["rate", "population"]:
            for T in [30, 50, 75]:
                spike_options = [10, 15, 20] if encoding == "rate" else [6, 10, 15]
                for max_spikes in spike_options:
                    delta_configs.append(
                        {
                            "feature_set": feature_set_name,
                            "encoding": encoding,
                            "T": T,
                            "max_spikes": max_spikes,
                            "epochs": 400,
                            "lr": 0.05,
                        }
                    )

    # Restrict snnTorch grid by default to keep runtime manageable.
    if args.snn_full_grid:
        snn_configs = delta_configs
    else:
        snn_configs = [
            {"feature_set": "all_7_norm", "encoding": "population", "T": 30, "max_spikes": 6, "epochs": args.snn_epochs, "lr": 1e-3},
            {"feature_set": "all_7_plus_symp_ratio", "encoding": "population", "T": 30, "max_spikes": 6, "epochs": args.snn_epochs, "lr": 1e-3},
            {"feature_set": "all_7_norm", "encoding": "rate", "T": 30, "max_spikes": 15, "epochs": args.snn_epochs, "lr": 1e-3},
        ]

    # Delta branch.
    delta_results = []
    best_delta = None
    for i, cfg in enumerate(delta_configs, start=1):
        X = norm[feature_sets[cfg["feature_set"]]].to_numpy(float)
        metrics, preds, cm, losses = run_spike_delta_loso(
            X=X,
            y=y,
            groups=groups,
            label_order=label_order,
            encoding=cfg["encoding"],
            T=cfg["T"],
            max_spikes=cfg["max_spikes"],
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            seed=100,
        )
        row = {
            **cfg,
            **metrics,
            "method": "delta",
            "config_id": i,
            "n_features": X.shape[1],
            "n_subjects": int(norm["subject_id"].nunique()),
        }
        delta_results.append(row)
        if best_delta is None or row["balanced_accuracy"] > best_delta[0]["balanced_accuracy"]:
            best_delta = (row, preds, cm, losses)

    delta_results_df = pd.DataFrame(delta_results).sort_values("balanced_accuracy", ascending=False)
    best_delta_row, best_delta_preds, best_delta_cm, best_delta_losses = best_delta

    # snnTorch branch.
    snn_results_df = pd.DataFrame()
    best_snn = None
    if not args.skip_snn:
        if not HAS_SNNTORCH:
            print("WARNING: snnTorch/PyTorch not available. Skipping snnTorch branch.")
            print(f"Import error: {TORCH_IMPORT_ERROR}")
        else:
            snn_results = []
            for i, cfg in enumerate(snn_configs, start=1):
                X = norm[feature_sets[cfg["feature_set"]]].to_numpy(float)
                metrics, preds, cm, losses = run_spike_snn_loso(
                    X=X,
                    y=y,
                    groups=groups,
                    label_order=label_order,
                    encoding=cfg["encoding"],
                    T=cfg["T"],
                    max_spikes=cfg["max_spikes"],
                    epochs=cfg["epochs"],
                    lr=cfg["lr"],
                    hidden=64,
                    beta=0.9,
                    seed=100,
                )
                row = {
                    **cfg,
                    **metrics,
                    "method": "snntorch",
                    "config_id": i,
                    "n_features": X.shape[1],
                    "n_subjects": int(norm["subject_id"].nunique()),
                }
                snn_results.append(row)
                if best_snn is None or row["balanced_accuracy"] > best_snn[0]["balanced_accuracy"]:
                    best_snn = (row, preds, cm, losses)

            snn_results_df = pd.DataFrame(snn_results).sort_values("balanced_accuracy", ascending=False)

    # Classical baselines on best delta feature set.
    best_feature_cols = feature_sets[best_delta_row["feature_set"]]
    X_best = norm[best_feature_cols].to_numpy(float)
    classical_df = run_classical_loso(X_best, y, groups)

    # Save delta outputs.
    delta_results_df.to_csv(output_dir / "spike_delta_config_results.csv", index=False)
    best_delta_preds.to_csv(output_dir / "best_spike_delta_predictions.csv", index=False)
    pd.DataFrame(best_delta_cm, index=label_order, columns=label_order).to_csv(output_dir / "best_spike_delta_confusion_matrix.csv")

    # Save snnTorch outputs if present.
    if best_snn is not None:
        best_snn_row, best_snn_preds, best_snn_cm, best_snn_losses = best_snn
        snn_results_df.to_csv(output_dir / "spike_snn_config_results.csv", index=False)
        best_snn_preds.to_csv(output_dir / "best_spike_snn_predictions.csv", index=False)
        pd.DataFrame(best_snn_cm, index=label_order, columns=label_order).to_csv(output_dir / "best_spike_snn_confusion_matrix.csv")
    else:
        best_snn_row = None

    classical_df.to_csv(output_dir / "classical_baselines_best_delta_feature_set.csv", index=False)

    summary = {
        "input_file": str(input_path),
        "n_rows": int(len(norm)),
        "n_subjects": int(norm["subject_id"].nunique()),
        "subjects": sorted(norm["subject_id"].unique().tolist()),
        "tasks": label_order,
        "raw_feature_cols": raw_feature_cols,
        "feature_sets": feature_sets,
        "best_spike_delta_config": best_delta_row,
        "best_spike_snn_config": best_snn_row,
        "classical_baselines_best_delta_feature_set": classical_df.to_dict(orient="records"),
        "snntorch_available": HAS_SNNTORCH,
        "note": "No data augmentation. Delta branch uses spike-count delta rule. snnTorch branch uses LIF neurons and surrogate-gradient training.",
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Plots.
    plot_top_configs(delta_results_df, output_dir / "top_spike_delta_configurations.png", "Top spike-delta configurations")
    plot_confusion_matrix(best_delta_cm, label_order, output_dir / "best_spike_delta_confusion_matrix.png", "Best spike-delta confusion matrix")
    plot_feature_heatmap(norm, best_feature_cols, output_dir / "best_delta_feature_set_heatmap.png")
    plot_example_spike_raster(
        X0=norm[best_feature_cols].to_numpy(float)[:1],
        encoding=best_delta_row["encoding"],
        T=int(best_delta_row["T"]),
        max_spikes=int(best_delta_row["max_spikes"]),
        output_path=output_dir / "best_delta_example_spike_raster.png",
        title="Example spike raster, best delta encoding",
    )
    plot_loss(best_delta_losses, output_dir / "best_spike_delta_training_loss.png", "Mean training loss, best delta config")

    if best_snn is not None:
        best_snn_row, best_snn_preds, best_snn_cm, best_snn_losses = best_snn
        best_snn_feature_cols = feature_sets[best_snn_row["feature_set"]]
        plot_top_configs(snn_results_df, output_dir / "top_spike_snn_configurations.png", "Top snnTorch SNN configurations")
        plot_confusion_matrix(best_snn_cm, label_order, output_dir / "best_spike_snn_confusion_matrix.png", "Best snnTorch SNN confusion matrix")
        plot_example_spike_raster(
            X0=norm[best_snn_feature_cols].to_numpy(float)[:1],
            encoding=best_snn_row["encoding"],
            T=int(best_snn_row["T"]),
            max_spikes=int(best_snn_row["max_spikes"]),
            output_path=output_dir / "best_snn_example_spike_raster.png",
            title="Example spike raster, best snnTorch encoding",
        )
        plot_loss(best_snn_losses, output_dir / "best_spike_snn_training_loss.png", "Mean training loss, best snnTorch config")

    # Zip outputs.
    zip_path = Path(str(output_dir) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in output_dir.glob("*"):
            z.write(p, arcname=f"{output_dir.name}/{p.name}")

    print("\nData summary")
    print(f"Rows: {len(norm)}")
    print(f"Subjects: {norm['subject_id'].nunique()}")
    print(f"Tasks: {label_order}")

    print("\nTop spike-delta results")
    print(delta_results_df.head(10).to_string(index=False))

    if best_snn is not None:
        print("\nTop snnTorch SNN results")
        print(snn_results_df.head(10).to_string(index=False))
    else:
        print("\nsnnTorch branch was skipped or unavailable.")

    print("\nClassical baselines on best delta feature set")
    print(classical_df.to_string(index=False))

    print("\nBest delta configuration")
    print(json.dumps(best_delta_row, indent=2, default=str))

    if best_snn is not None:
        print("\nBest snnTorch configuration")
        print(json.dumps(best_snn_row, indent=2, default=str))

    print(f"\nOutputs saved to: {output_dir}")
    print(f"Zipped outputs: {zip_path}")


if __name__ == "__main__":
    main()

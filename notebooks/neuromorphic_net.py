#!/usr/bin/env python3
"""
Spike-based neuromorphic-style classification of EDA features.

This script:
1. Loads EDA feature data from CSV.
2. Treats Sample as subject identifier.
3. Uses four classes: rest0, task1, task2, task3.
4. Performs within-subject min-max normalization.
5. Adds a derived EDASymp ratio feature.
6. Encodes normalized features into spikes using:
   - deterministic rate encoding
   - Gaussian population encoding
7. Trains a compact spike-count delta-rule readout.
8. Evaluates with leave-one-subject-out cross-validation.
9. Compares against classical baselines.
10. Saves metrics, predictions, confusion matrices, and plots.

Important:
This is not data augmentation.
This uses spike encoding plus a delta-rule softmax readout trained on spike counts.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC


# ---------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------

def deterministic_rate_encode(
    X: np.ndarray,
    T: int = 50,
    max_spikes: int = 15,
) -> np.ndarray:
    """
    Deterministic rate encoding.

    Parameters
    ----------
    X:
        Feature matrix of shape [n_samples, n_features], values in [0, 1].
    T:
        Number of spike time steps.
    max_spikes:
        Maximum number of spikes per feature.

    Returns
    -------
    S:
        Binary spike tensor of shape [n_samples, T, n_features].
    """
    X = np.asarray(X, dtype=float)

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input feature matrix contains NaN or inf.")

    if X.min() < -1e-8 or X.max() > 1 + 1e-8:
        raise ValueError(f"Input features must be in [0, 1]. Got min={X.min()}, max={X.max()}.")

    S = np.zeros((X.shape[0], T, X.shape[1]), dtype=np.float32)

    for i in range(X.shape[0]):
        for f in range(X.shape[1]):
            n_spikes = int(np.round(X[i, f] * max_spikes))

            if n_spikes > 0:
                spike_times = np.linspace(0, T - 1, n_spikes).astype(int)
                S[i, spike_times, f] = 1.0

    return S


def gaussian_population_encode(
    X: np.ndarray,
    T: int = 50,
    max_spikes: int = 10,
    n_pop: int = 3,
    sigma: float = 0.25,
) -> np.ndarray:
    """
    Gaussian population encoding.

    Each scalar feature is expanded into n_pop Gaussian-tuned channels.
    This gives the spike model a richer representation than one spike train per feature.
    """
    X = np.asarray(X, dtype=float)

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input feature matrix contains NaN or inf.")

    centers = np.linspace(0, 1, n_pop)
    responses = []

    for f in range(X.shape[1]):
        vals = X[:, f:f + 1]
        r = np.exp(-0.5 * ((vals - centers[None, :]) / sigma) ** 2)
        r = r / (r.max(axis=1, keepdims=True) + 1e-8)
        responses.append(r)

    X_pop = np.concatenate(responses, axis=1).astype(np.float32)

    return deterministic_rate_encode(X_pop, T=T, max_spikes=max_spikes)


# ---------------------------------------------------------------------
# Delta-rule readout
# ---------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    """Stable softmax."""
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
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Train a compact spike-count readout using a delta-rule/softmax update.

    The input spike tensor is first reduced to spike counts per input channel.
    The class readout is then trained using a multiclass delta-rule equivalent
    to gradient descent on softmax cross-entropy.
    """
    rng = np.random.default_rng(seed)

    X_counts = S_train.sum(axis=1).astype(float)

    # Normalize spike counts per input channel for numerical stability.
    X_counts = X_counts / (X_counts.max(axis=0, keepdims=True) + 1e-8)

    W = rng.normal(0, 0.05, size=(X_counts.shape[1], n_classes))
    b = np.zeros(n_classes)

    Y = np.eye(n_classes)[y_train]

    losses = []

    for _ in range(epochs):
        logits = X_counts @ W + b
        P = softmax(logits)

        loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1)) + l2 * np.sum(W * W)

        grad_logits = (P - Y) / len(y_train)
        grad_W = X_counts.T @ grad_logits + 2 * l2 * W
        grad_b = grad_logits.sum(axis=0)

        W -= lr * grad_W
        b -= lr * grad_b

        losses.append(float(loss))

    return W, b, losses


def predict_delta_readout(
    S: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict classes from spike counts using trained readout weights.
    """
    X_counts = S.sum(axis=1).astype(float)
    X_counts = X_counts / (X_counts.max(axis=0, keepdims=True) + 1e-8)

    logits = X_counts @ W + b
    y_pred = logits.argmax(axis=1)

    return y_pred, logits


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def run_spike_delta_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_order: list[str],
    encoding: str = "population",
    T: int = 30,
    max_spikes: int = 6,
    epochs: int = 400,
    lr: float = 0.05,
    seed: int = 100,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """Leave‑one‑subject‑out evaluation for spike encoding plus delta‑rule readout."""
    all_true = []
    all_pred = []
    rows = []
    fold_losses = []

    logo = LeaveOneGroupOut()

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        if train_subjects.intersection(test_subjects):
            raise RuntimeError("Subject leakage detected.")

        # Encode spikes according to the chosen scheme
        if encoding == "rate":
            S_train = deterministic_rate_encode(X[train_idx], T=T, max_spikes=max_spikes)
            S_test = deterministic_rate_encode(X[test_idx], T=T, max_spikes=max_spikes)
        elif encoding == "population":
            S_train = gaussian_population_encode(X[train_idx], T=T, max_spikes=max_spikes)
            S_test = gaussian_population_encode(X[test_idx], T=T, max_spikes=max_spikes)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # Train delta‑rule readout
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
    return metrics, pd.DataFrame(rows), cm, np.asarray(fold_losses)


def run_spike_snn_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_order: list[str],
    encoding: str = "population",
    T: int = 30,
    max_spikes: int = 6,
    epochs: int = 200,
    lr: float = 1e-3,
    hidden: int = 128,
    beta: float = 0.9,
    seed: int = 100,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """Leave‑one‑subject‑out evaluation for a spiking neural network (snnTorch)."""
    all_true = []
    all_pred = []
    rows = []
    fold_losses = []

    logo = LeaveOneGroupOut()

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        if train_subjects.intersection(test_subjects):
            raise RuntimeError("Subject leakage detected.")

        # Encode spikes
        if encoding == "rate":
            S_train = deterministic_rate_encode(X[train_idx], T=T, max_spikes=max_spikes)
            S_test = deterministic_rate_encode(X[test_idx], T=T, max_spikes=max_spikes)
        elif encoding == "population":
            S_train = gaussian_population_encode(X[train_idx], T=T, max_spikes=max_spikes)
            S_test = gaussian_population_encode(X[test_idx], T=T, max_spikes=max_spikes)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

        # Train SNN
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
    return metrics, pd.DataFrame(rows), cm, np.asarray(fold_losses)


def run_classical_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """
    Classical LOSO baselines on the same normalized features.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVM": SVC(kernel="linear", class_weight="balanced"),
        "RBFSVM": SVC(kernel="rbf", class_weight="balanced"),
    }

    rows = []

    for name, clf in models.items():
        all_true = []
        all_pred = []

        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])

            all_true.extend(y[test_idx].tolist())
            all_pred.extend(pred.tolist())

        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(all_true, all_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)),
            "macro_f1": float(f1_score(all_true, all_pred, average="macro")),
            "weighted_f1": float(f1_score(all_true, all_pred, average="weighted")),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_top_configs(results_df: pd.DataFrame, output_path: Path) -> None:
    top = results_df.sort_values("balanced_accuracy", ascending=False).head(12).copy()

    labels = [
        f"{r['feature_set']} | {r['encoding']} | T={int(r['T'])} | sp={int(r['max_spikes'])}"
        for _, r in top.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ypos = np.arange(len(top))
    vals = top["balanced_accuracy"].values

    ax.barh(ypos, vals, color="steelblue")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Top spike-delta configurations, LOSO")

    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 0.95), i, f"{v:.2f}", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, label_order: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_order,
        yticklabels=label_order,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Best spike-delta confusion matrix")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_heatmap(norm_df: pd.DataFrame, feature_cols: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        norm_df[feature_cols].to_numpy(),
        vmin=0,
        vmax=1,
        cmap="viridis",
        xticklabels=feature_cols,
        yticklabels=False,
        ax=ax,
    )

    ax.set_title("Normalized features for best feature set")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_example_spike_raster(
    X0: np.ndarray,
    encoding: str,
    T: int,
    max_spikes: int,
    output_path: Path,
) -> None:
    if encoding == "population":
        S0 = gaussian_population_encode(X0, T=T, max_spikes=max_spikes)[0]
    else:
        S0 = deterministic_rate_encode(X0, T=T, max_spikes=max_spikes)[0]

    time_idx, channel_idx = np.where(S0.T == 1)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.scatter(time_idx, channel_idx, marker="|", s=40)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Input channel")
    ax.set_title("Example spike raster, best encoding")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eda_features.csv")
    parser.add_argument("--output", default="more_features_spike_delta_outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Load data
    raw = pd.read_csv(input_path)

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

    feature_cols = [
        c for c in raw.columns
        if c not in ["Sample", "subject_id", "Task"]
        and pd.api.types.is_numeric_dtype(raw[c])
    ]

    # Within-subject min-max normalization
    norm = raw.copy()

    for c in feature_cols:
        mn = norm.groupby("subject_id")[c].transform("min")
        mx = norm.groupby("subject_id")[c].transform("max")
        norm[c + "_norm"] = ((norm[c] - mn) / (mx - mn + 1e-8)).clip(0, 1)

    # Derived normalized EDASymp ratio
    norm["EDASymp_Ratio"] = norm["EDASymp Power"] / (norm["Total Power"] + 1e-12)

    mn = norm.groupby("subject_id")["EDASymp_Ratio"].transform("min")
    mx = norm.groupby("subject_id")["EDASymp_Ratio"].transform("max")
    norm["EDASymp_Ratio_norm"] = ((norm["EDASymp_Ratio"] - mn) / (mx - mn + 1e-8)).clip(0, 1)

    feature_sets = {
        "all_7_norm": [c + "_norm" for c in feature_cols],
        "all_7_plus_symp_ratio": [c + "_norm" for c in feature_cols] + ["EDASymp_Ratio_norm"],
        "no_count_plus_symp_ratio": [
            c + "_norm" for c in feature_cols
            if c != "Num Peaks"
        ] + ["EDASymp_Ratio_norm"],
    }

    label_to_int = {lab: i for i, lab in enumerate(label_order)}
    y = norm["Task"].map(label_to_int).to_numpy(np.int64)
    groups = norm["subject_id"].to_numpy()

    # Save normalized features
    norm.to_csv(output_dir / "features_normalized_with_ratio.csv", index=False)

    configs = []

    for feature_set_name in feature_sets:
        for encoding in ["rate", "population"]:
            for T in [30, 50, 75]:
                if encoding == "rate":
                    spike_options = [10, 15, 20]
                else:
                    spike_options = [6, 10, 15]

                for max_spikes in spike_options:
                    configs.append({
                        "feature_set": feature_set_name,
                        "encoding": encoding,
                        "T": T,
                        "max_spikes": max_spikes,
                        "epochs": 400,
                        "lr": 0.05,
                    })

    # -----------------------------------------------------------------
    # Run configurations for delta‑rule readout and spiking neural network
    # -----------------------------------------------------------------
    delta_results = []
    snn_results = []
    best_delta = None
    best_snn = None

    for i, cfg in enumerate(configs, start=1):
        feature_cols_this = feature_sets[cfg["feature_set"]]
        X = norm[feature_cols_this].to_numpy(float)

        # ----- Delta‑rule readout -----
        delta_metrics, delta_preds, delta_cm, delta_losses = run_spike_delta_loso(
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

        delta_row = {
            **cfg,
            **delta_metrics,
            "method": "delta",
            "config_id": i,
            "n_features": X.shape[1],
            "n_subjects": int(norm["subject_id"].nunique()),
        }
        delta_results.append(delta_row)
        if best_delta is None or delta_row["balanced_accuracy"] > best_delta[0]["balanced_accuracy"]:
            best_delta = (delta_row, delta_preds, delta_cm, delta_losses)

        # ----- Spiking Neural Network (snnTorch) -----
        # Use a smaller epoch count for SNN to keep runtime reasonable.
        snn_metrics, snn_preds, snn_cm, snn_losses = run_spike_snn_loso(
            X=X,
            y=y,
            groups=groups,
            label_order=label_order,
            encoding=cfg["encoding"],
            T=cfg["T"],
            max_spikes=cfg["max_spikes"],
            epochs=200,  # fewer epochs than delta‑rule by default
            lr=1e-3,
            hidden=128,
            beta=0.9,
            seed=100,
        )

        snn_row = {
            **cfg,
            **snn_metrics,
            "method": "snn",
            "config_id": i,
            "n_features": X.shape[1],
            "n_subjects": int(norm["subject_id"].nunique()),
        }
        snn_results.append(snn_row)
        if best_snn is None or snn_row["balanced_accuracy"] > best_snn[0]["balanced_accuracy"]:
            best_snn = (snn_row, snn_preds, snn_cm, snn_losses)

    # DataFrames for each method
    delta_results_df = pd.DataFrame(delta_results).sort_values("balanced_accuracy", ascending=False)
    snn_results_df = pd.DataFrame(snn_results).sort_values("balanced_accuracy", ascending=False)

    # Best configurations per method
    best_delta_row, best_delta_preds, best_delta_cm, best_delta_losses = best_delta
    best_snn_row, best_snn_preds, best_snn_cm, best_snn_losses = best_snn
    best_feature_cols = feature_sets[best_row["feature_set"]]
    X_best = norm[best_feature_cols].to_numpy(float)

    classical_df = run_classical_loso(
        X=X_best,
        y=y,
        groups=groups,
    )

    # -----------------------------------------------------------------
    # Save outputs for both delta‑rule and SNN methods
    # -----------------------------------------------------------------
    # Delta‑rule results (original behavior)
    delta_results_df.to_csv(output_dir / "spike_delta_config_results.csv", index=False)
    best_delta_preds.to_csv(output_dir / "best_spike_delta_predictions.csv", index=False)
    pd.DataFrame(
        best_delta_cm,
        index=label_order,
        columns=label_order,
    ).to_csv(output_dir / "best_spike_delta_confusion_matrix.csv")

    # SNN results
    snn_results_df.to_csv(output_dir / "spike_snn_config_results.csv", index=False)
    best_snn_preds.to_csv(output_dir / "best_spike_snn_predictions.csv", index=False)
    pd.DataFrame(
        best_snn_cm,
        index=label_order,
        columns=label_order,
    ).to_csv(output_dir / "best_spike_snn_confusion_matrix.csv")

    # Classical baselines (unchanged)
    classical_df.to_csv(output_dir / "classical_baselines_best_feature_set.csv", index=False)

    # -----------------------------------------------------------------
    # Summary JSON (includes both best configurations)
    # -----------------------------------------------------------------
    summary = {
        "input_file": str(input_path),
        "n_rows": int(len(norm)),
        "n_subjects": int(norm["subject_id"].nunique()),
        "subjects": sorted(norm["subject_id"].unique().tolist()),
        "tasks": label_order,
        "raw_feature_cols": feature_cols,
        "feature_sets": feature_sets,
        "best_spike_delta_config": best_delta_row,
        "best_spike_snn_config": best_snn_row,
        "classical_baselines_best_feature_set": classical_df.to_dict(orient="records"),
        "note": (
            "This script evaluates spike‑encoded features using a simple delta‑rule readout "
            "and a spiking neural network (snnTorch). No data augmentation was used."
        ),
    }

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # -----------------------------------------------------------------
    # Plots for delta‑rule method (kept for backward compatibility)
    # -----------------------------------------------------------------
    plot_top_configs(delta_results_df, output_dir / "top_spike_delta_configurations.png")
    plot_confusion_matrix(best_delta_cm, label_order, output_dir / "best_spike_delta_confusion_matrix.png")
    plot_feature_heatmap(norm, best_feature_cols, output_dir / "best_feature_set_heatmap.png")

    plot_example_spike_raster(
        X0=norm[best_feature_cols].to_numpy(float)[:1],
        encoding=best_delta_row["encoding"],
        T=int(best_delta_row["T"]),
        max_spikes=int(best_delta_row["max_spikes"]),
        output_path=output_dir / "best_encoding_example_spike_raster.png",
    )

    # Training loss plot for delta‑rule
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(best_delta_losses.mean(axis=0))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross‑entropy loss")
    ax.set_title("Mean training loss across LOSO folds, best delta‑rule config")
    fig.tight_layout()
    fig.savefig(output_dir / "best_spike_delta_training_loss.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Additional plots for the SNN method
    # -----------------------------------------------------------------
    plot_top_configs(snn_results_df, output_dir / "top_spike_snn_configurations.png")
    plot_confusion_matrix(best_snn_cm, label_order, output_dir / "best_spike_snn_confusion_matrix.png")
    plot_example_spike_raster(
        X0=norm[best_feature_cols].to_numpy(float)[:1],
        encoding=best_snn_row["encoding"],
        T=int(best_snn_row["T"]),
        max_spikes=int(best_snn_row["max_spikes"]),
        output_path=output_dir / "best_spike_snn_example_spike_raster.png",
    )
    # Training loss plot for SNN
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(best_snn_losses.mean(axis=0))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross‑entropy loss")
    ax.set_title("Mean training loss across LOSO folds, best SNN config")
    fig.tight_layout()
    fig.savefig(output_dir / "best_spike_snn_training_loss.png", dpi=200)
    plt.close(fig)

    # Zip outputs
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
    print(results_df.head(10).to_string(index=False))

    print("\nClassical baselines on best feature set")
    print(classical_df.to_string(index=False))

    print("\nBest configuration")
    print(json.dumps(best_row, indent=2, default=str))

    print(f"\nOutputs saved to: {output_dir}")
    print(f"Zipped outputs: {zip_path}")


if __name__ == "__main__":
    main()
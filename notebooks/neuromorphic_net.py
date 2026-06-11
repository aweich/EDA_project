#!/usr/bin/env python3
"""
EDA feature encoding and compact SNN training/testing without data augmentation.

Pipeline:
1. Load EDA feature table.
2. Detect numeric feature columns.
3. Normalize features:
   - within-subject min-max if subject_id exists
   - global min-max if subject_id is absent
4. Encode normalized features into deterministic rate-coded spike trains.
5. Train/test a compact single-layer LIF SNN.
   - If subject_id exists: leave-one-subject-out cross-validation.
   - If subject_id does not exist: overfit sanity check only.
6. Save metrics, predictions, spike rasters, training curves, and confusion matrices.

Important:
If the input CSV has only one row per class and no subject_id column, this script cannot
estimate generalization. It can only verify that encoding and SNN training work technically.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_LABEL_COL = "Task"
DEFAULT_SUBJECT_COL = "subject_id"


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_feature_table(path: str, label_col: str = DEFAULT_LABEL_COL) -> pd.DataFrame:
    """Load feature CSV and validate the label column."""
    df = pd.read_csv(path)

    if label_col not in df.columns:
        raise ValueError(
            f"Expected label column '{label_col}', but found columns: {list(df.columns)}"
        )

    return df


def get_numeric_feature_columns(
    df: pd.DataFrame,
    label_col: str = DEFAULT_LABEL_COL,
    subject_col: str = DEFAULT_SUBJECT_COL,
) -> List[str]:
    """Return numeric feature columns, excluding label and subject columns."""
    excluded = {label_col, subject_col}
    feature_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found.")

    return feature_cols


def encode_labels(labels: pd.Series) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """Convert string labels to integer labels."""
    class_names = list(pd.unique(labels.astype(str)))
    label_to_int = {name: i for i, name in enumerate(class_names)}
    int_to_label = {i: name for name, i in label_to_int.items()}
    y = labels.astype(str).map(label_to_int).to_numpy(dtype=np.int64)
    return y, label_to_int, int_to_label


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def minmax_normalize_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    subject_col: str = DEFAULT_SUBJECT_COL,
    epsilon: float = 1e-8,
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Normalize feature columns to [0, 1].

    If subject_id exists:
        Normalization is performed within each subject across that subject's rows.

    If subject_id does not exist:
        Normalization is performed globally across all rows.
        This is only a fallback for tiny example files.
    """
    out = df.copy()
    norm_cols = []

    if subject_col in out.columns:
        mode = "within_subject_minmax"
        grouped = out.groupby(subject_col, sort=False)

        for col in feature_cols:
            mn = grouped[col].transform("min")
            mx = grouped[col].transform("max")
            norm_col = f"{col}_norm"
            out[norm_col] = (out[col] - mn) / (mx - mn + epsilon)
            out[norm_col] = out[norm_col].clip(0.0, 1.0)
            norm_cols.append(norm_col)

    else:
        mode = "global_minmax_no_subject_id"
        print(
            "WARNING: no subject_id column found. "
            "Using global min-max normalization across all rows. "
            "This is not valid for subject-independent evaluation."
        )

        for col in feature_cols:
            mn = out[col].min()
            mx = out[col].max()
            norm_col = f"{col}_norm"
            out[norm_col] = (out[col] - mn) / (mx - mn + epsilon)
            out[norm_col] = out[norm_col].clip(0.0, 1.0)
            norm_cols.append(norm_col)

    if not np.isfinite(out[norm_cols].to_numpy()).all():
        raise ValueError("Normalized features contain NaN or inf.")

    return out, norm_cols, mode


# ---------------------------------------------------------------------
# Spike encoding
# ---------------------------------------------------------------------

def deterministic_rate_encode(
    X: np.ndarray,
    T: int = 50,
    max_spikes: int = 15,
) -> np.ndarray:
    """
    Deterministic rate encoding.

    Input:
        X: [n_samples, n_features], values in [0, 1]

    Output:
        spikes: [n_samples, T, n_features], binary spike tensor

    Each feature value controls the number of spikes.
    A value of 0 gives 0 spikes.
    A value of 1 gives max_spikes spikes.
    """
    X = np.asarray(X, dtype=np.float32)

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN or inf.")

    if X.min() < -1e-8 or X.max() > 1 + 1e-8:
        raise ValueError(f"X must be in [0, 1]. Got min={X.min()}, max={X.max()}.")

    if T <= 0:
        raise ValueError("T must be > 0.")

    if max_spikes < 0:
        raise ValueError("max_spikes must be >= 0.")

    n_samples, n_features = X.shape
    spikes = np.zeros((n_samples, T, n_features), dtype=np.float32)

    for i in range(n_samples):
        for f in range(n_features):
            n_spikes = int(np.round(X[i, f] * max_spikes))

            if n_spikes > 0:
                spike_times = np.linspace(0, T - 1, n_spikes).astype(int)
                spikes[i, spike_times, f] = 1.0

    return spikes


# ---------------------------------------------------------------------
# SNN model
# ---------------------------------------------------------------------

class SurrogateSpike(torch.autograd.Function):
    """
    Binary spike function with fast-sigmoid surrogate gradient.

    Forward:
        spike = 1 if membrane_input > 0 else 0

    Backward:
        approximate derivative using 1 / (1 + abs(x))^2
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (input_tensor,) = ctx.saved_tensors
        surrogate_grad = 1.0 / (1.0 + torch.abs(input_tensor)) ** 2
        return grad_output * surrogate_grad


class SingleLayerLIFSNN(nn.Module):
    """
    Compact single-layer LIF SNN.

    Input:
        spikes [batch, T, n_features]

    Output:
        spike_counts [batch, n_classes]

    The model predicts the class with the highest output spike count.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        beta: float = 0.9,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)
        self.beta = beta
        self.threshold = threshold

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        batch_size, T, _ = spikes.shape

        mem = torch.zeros(batch_size, self.fc.out_features, device=spikes.device)
        spike_count = torch.zeros_like(mem)

        for t in range(T):
            current = self.fc(spikes[:, t, :])
            mem = self.beta * mem + current

            out_spike = SurrogateSpike.apply(mem - self.threshold)

            mem = mem * (1.0 - out_spike.detach())
            spike_count = spike_count + out_spike

        return spike_count


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def train_one_model(
    X_spikes_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    beta: float = 0.9,
    threshold: float = 1.0,
    seed: int = 42,
) -> SingleLayerLIFSNN:
    """Train one SNN model."""
    set_seeds(seed)

    n_features = X_spikes_train.shape[2]

    model = SingleLayerLIFSNN(
        n_features=n_features,
        n_classes=n_classes,
        beta=beta,
        threshold=threshold,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_spikes_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)

        loss.backward()
        optimizer.step()

    return model


def predict_model(model: SingleLayerLIFSNN, X_spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Predict labels and return output spike counts."""
    model.eval()

    X_t = torch.tensor(X_spikes, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        spike_counts = logits.cpu().numpy()

    return y_pred, spike_counts


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "n_predictions": int(len(y_true)),
    }


def run_loso_evaluation(
    X_norm: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    class_names: List[str],
    T: int,
    max_spikes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Run leave-one-subject-out evaluation.

    For each held-out subject:
        train on all other subjects
        test on held-out subject
    """
    logo = LeaveOneGroupOut()

    all_true = []
    all_pred = []
    prediction_rows = []

    n_classes = len(class_names)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_norm, y, subject_ids), start=1):
        train_subjects = set(subject_ids[train_idx])
        test_subjects = set(subject_ids[test_idx])

        overlap = train_subjects.intersection(test_subjects)
        if len(overlap) > 0:
            raise RuntimeError(f"Subject leakage detected in fold {fold_idx}: {overlap}")

        X_train = X_norm[train_idx]
        y_train = y[train_idx]
        X_test = X_norm[test_idx]
        y_test = y[test_idx]

        X_spikes_train = deterministic_rate_encode(X_train, T=T, max_spikes=max_spikes)
        X_spikes_test = deterministic_rate_encode(X_test, T=T, max_spikes=max_spikes)

        model = train_one_model(
            X_spikes_train=X_spikes_train,
            y_train=y_train,
            n_classes=n_classes,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=42 + fold_idx,
        )

        y_pred, spike_counts = predict_model(model, X_spikes_test)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        for i, sample_idx in enumerate(test_idx):
            row = {
                "fold": fold_idx,
                "subject_id": subject_ids[sample_idx],
                "y_true": int(y_test[i]),
                "y_pred": int(y_pred[i]),
                "true_class": class_names[int(y_test[i])],
                "pred_class": class_names[int(y_pred[i])],
            }

            for c_idx, c_name in enumerate(class_names):
                row[f"out_spikes_{c_name}"] = float(spike_counts[i, c_idx])

            prediction_rows.append(row)

    y_true_arr = np.asarray(all_true)
    y_pred_arr = np.asarray(all_pred)

    metrics = evaluate_predictions(y_true_arr, y_pred_arr)
    metrics_df = pd.DataFrame([metrics])
    predictions_df = pd.DataFrame(prediction_rows)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=np.arange(len(class_names)))

    return metrics_df, predictions_df, cm


def run_tiny_overfit_sanity_check(
    X_norm: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    T: int,
    max_spikes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[float], List[float]]:
    """
    Run an overfit sanity check when no subject_id column exists.

    This trains and evaluates on the same samples.
    This is NOT a valid performance estimate.
    """
    X_spikes = deterministic_rate_encode(X_norm, T=T, max_spikes=max_spikes)

    n_classes = len(class_names)
    n_features = X_spikes.shape[2]

    model = SingleLayerLIFSNN(n_features=n_features, n_classes=n_classes)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_spikes, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    losses = []
    accuracies = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(X_t)
        loss = criterion(logits, y_t)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y_t).float().mean().item()

        losses.append(float(loss.item()))
        accuracies.append(float(acc))

    y_pred, spike_counts = predict_model(model, X_spikes)

    metrics = evaluate_predictions(y, y_pred)
    metrics["note"] = "overfit_sanity_check_only"
    metrics_df = pd.DataFrame([metrics])

    prediction_rows = []
    for i in range(len(y)):
        row = {
            "sample_index": i,
            "y_true": int(y[i]),
            "y_pred": int(y_pred[i]),
            "true_class": class_names[int(y[i])],
            "pred_class": class_names[int(y_pred[i])],
        }

        for c_idx, c_name in enumerate(class_names):
            row[f"out_spikes_{c_name}"] = float(spike_counts[i, c_idx])

        prediction_rows.append(row)

    predictions_df = pd.DataFrame(prediction_rows)
    cm = confusion_matrix(y, y_pred, labels=np.arange(len(class_names)))

    return metrics_df, predictions_df, cm, losses, accuracies


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str,
) -> None:
    """Save confusion matrix heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_curve(
    losses: List[float],
    accuracies: List[float],
    output_path: Path,
) -> None:
    """Save training loss and accuracy curve."""
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(losses, label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")

    ax2 = ax1.twinx()
    ax2.plot(accuracies, color="orange", label="Training accuracy")
    ax2.set_ylabel("Training accuracy")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")

    plt.title("SNN training curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_spike_raster(
    spikes: np.ndarray,
    feature_names: List[str],
    output_path: Path,
    title: str = "Example input spike raster",
) -> None:
    """
    Save a raster plot for one sample.

    spikes shape:
        [T, n_features]
    """
    T, n_features = spikes.shape

    time_idx, feature_idx = np.where(spikes.T == 1)

    plt.figure(figsize=(10, 4))
    plt.scatter(time_idx, feature_idx, marker="|", s=40)
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Time step")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_normalized_features(
    norm_df: pd.DataFrame,
    norm_cols: List[str],
    label_col: str,
    output_path: Path,
) -> None:
    """Save heatmap of normalized features."""
    mat = norm_df[norm_cols].to_numpy()
    labels = norm_df[label_col].astype(str).to_list()

    plt.figure(figsize=(10, max(4, len(labels) * 0.25)))
    sns.heatmap(
        mat,
        cmap="viridis",
        vmin=0,
        vmax=1,
        yticklabels=labels,
        xticklabels=[c.replace("_norm", "") for c in norm_cols],
    )
    plt.title("Normalized features used for spike encoding")
    plt.xlabel("Feature")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="eda_features.csv")
    parser.add_argument("--output_dir", type=str, default="snn_no_aug_outputs")
    parser.add_argument("--label_col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--subject_col", type=str, default=DEFAULT_SUBJECT_COL)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--max_spikes", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_feature_table(args.input_csv, label_col=args.label_col)
    feature_cols = get_numeric_feature_columns(
        df,
        label_col=args.label_col,
        subject_col=args.subject_col,
    )

    y, label_to_int, int_to_label = encode_labels(df[args.label_col])
    class_names = [int_to_label[i] for i in range(len(int_to_label))]

    # Normalize
    norm_df, norm_cols, normalization_mode = minmax_normalize_features(
        df,
        feature_cols=feature_cols,
        subject_col=args.subject_col,
    )

    X_norm = norm_df[norm_cols].to_numpy(dtype=np.float32)

    # Save normalized features
    norm_df.to_csv(output_dir / "features_normalized.csv", index=False)

    # Encode all samples once for plotting
    all_spikes = deterministic_rate_encode(
        X_norm,
        T=args.T,
        max_spikes=args.max_spikes,
    )

    np.save(output_dir / "spikes_deterministic_rate.npy", all_spikes)

    spike_info = {
        "spike_shape": list(all_spikes.shape),
        "T": args.T,
        "max_spikes": args.max_spikes,
        "n_features": len(feature_cols),
        "n_classes": len(class_names),
        "feature_cols": feature_cols,
        "norm_cols": norm_cols,
        "class_names": class_names,
        "normalization_mode": normalization_mode,
    }

    with open(output_dir / "spike_info.json", "w") as f:
        json.dump(spike_info, f, indent=2)

    # Train/test
    if args.subject_col in norm_df.columns:
        print("Running leave-one-subject-out evaluation.")

        subject_ids = norm_df[args.subject_col].to_numpy()

        metrics_df, predictions_df, cm = run_loso_evaluation(
            X_norm=X_norm,
            y=y,
            subject_ids=subject_ids,
            class_names=class_names,
            T=args.T,
            max_spikes=args.max_spikes,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            output_dir=output_dir,
        )

        metrics_df["evaluation"] = "leave_one_subject_out"

        plot_confusion_matrix(
            cm,
            class_names=class_names,
            output_path=output_dir / "confusion_matrix_loso.png",
            title="SNN LOSO confusion matrix",
        )

    else:
        print(
            "No subject_id column found. "
            "Running overfit sanity check only. "
            "This is not a valid performance estimate."
        )

        metrics_df, predictions_df, cm, losses, accuracies = run_tiny_overfit_sanity_check(
            X_norm=X_norm,
            y=y,
            class_names=class_names,
            T=args.T,
            max_spikes=args.max_spikes,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        metrics_df["evaluation"] = "overfit_sanity_check_only"

        plot_training_curve(
            losses,
            accuracies,
            output_path=output_dir / "training_curve_overfit_sanity_check.png",
        )

        plot_confusion_matrix(
            cm,
            class_names=class_names,
            output_path=output_dir / "confusion_matrix_overfit_sanity_check.png",
            title="SNN confusion matrix, overfit sanity check only",
        )

    # Save metrics and predictions
    metrics_df.to_csv(output_dir / "snn_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "snn_predictions.csv", index=False)

    # Plots
    plot_spike_raster(
        all_spikes[0],
        feature_names=feature_cols,
        output_path=output_dir / "example_spike_raster.png",
        title=f"Example spike raster: {df[args.label_col].iloc[0]}",
    )

    plot_normalized_features(
        norm_df,
        norm_cols=norm_cols,
        label_col=args.label_col,
        output_path=output_dir / "normalized_feature_heatmap.png",
    )

    # Summary
    summary = {
        "input_csv": args.input_csv,
        "output_dir": str(output_dir),
        "normalization_mode": normalization_mode,
        "n_samples": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(class_names)),
        "class_names": class_names,
        "feature_cols": feature_cols,
        "spike_shape": list(all_spikes.shape),
        "evaluation": metrics_df["evaluation"].iloc[0],
        "metrics": metrics_df.to_dict(orient="records"),
        "warning": (
            "If no subject_id column was present, this run is only an overfit sanity check "
            "and must not be interpreted as generalization performance."
        ),
    }

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
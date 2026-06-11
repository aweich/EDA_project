#!/usr/bin/env python3
"""
Compare EDA feature groups using a fixed best snnTorch SNN configuration,
classical machine-learning baselines, and a spike-count delta-rule readout.

Feature groups
--------------
Tonic:
    Mean Tonic, Max Slope
Phasic:
    Num Peaks, Mean Amplitude, Std Amplitude
PSD:
    Total Power, EDASymp Power
All:
    Mean Tonic, Max Slope, Num Peaks, Mean Amplitude, Std Amplitude,
    Total Power, EDASymp Power

Default fixed SNN configuration
-------------------------------
From the user's best tuning result:
    hidden=64
    beta=0.95
    epochs=100
    lr=0.005
    encoding=population
    T=30
    max_spikes=10
    n_pop=3
    sigma=0.25

Important
---------
The default Gaussian population encoding uses max-normalized Gaussian channel
responses, because this matches the previous tuning script and the reported best
configuration. Use --raw_gaussian if you explicitly want raw Gaussian responses.
If --raw_gaussian is used, the previous best hyperparameters may no longer be optimal.

Run example
-----------
python compare_eda_feature_groups_snn_delta_ml.py \
  --input test/eda_features_29_samples.csv \
  --output feature_group_comparison \
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
# Data loading and feature grouping
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


def required_raw_feature_groups() -> Dict[str, List[str]]:
    return {
        "tonic": ["Mean Tonic", "Max Slope"],
        "phasic": ["Num Peaks", "Mean Amplitude", "Std Amplitude"],
        "psd": ["Total Power", "EDASymp Power"],
        "all": [
            "Mean Tonic", "Max Slope", "Num Peaks", "Mean Amplitude",
            "Std Amplitude", "Total Power", "EDASymp Power",
        ],
    }


def prepare_data(args):
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

    raw_groups = required_raw_feature_groups()
    missing = sorted(set(raw_groups["all"]).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required raw feature columns: {missing}")

    norm = df.copy()
    feature_groups_norm = {}
    for group_name, raw_cols in raw_groups.items():
        norm_cols = []
        for col in raw_cols:
            norm_col = col + "_norm"
            if args.use_existing_norm and norm_col in norm.columns:
                pass
            else:
                mn = norm.groupby("subject_id")[col].transform("min")
                mx = norm.groupby("subject_id")[col].transform("max")
                norm[norm_col] = ((norm[col] - mn) / (mx - mn + 1e-8)).clip(0, 1)
            norm_cols.append(norm_col)
        feature_groups_norm[group_name] = norm_cols

    if args.include_ratio and {"Total Power", "EDASymp Power"}.issubset(norm.columns):
        norm["EDASymp_Ratio"] = norm["EDASymp Power"] / (norm["Total Power"] + 1e-12)
        mn = norm.groupby("subject_id")["EDASymp_Ratio"].transform("min")
        mx = norm.groupby("subject_id")["EDASymp_Ratio"].transform("max")
        norm["EDASymp_Ratio_norm"] = ((norm["EDASymp_Ratio"] - mn) / (mx - mn + 1e-8)).clip(0, 1)
        feature_groups_norm["psd"] = feature_groups_norm["psd"] + ["EDASymp_Ratio_norm"]
        feature_groups_norm["all"] = feature_groups_norm["all"] + ["EDASymp_Ratio_norm"]

    label_to_int = {lab: i for i, lab in enumerate(label_order)}
    y = norm["Task"].map(label_to_int).to_numpy(np.int64)
    groups = norm["subject_id"].to_numpy()

    for group_name, cols in feature_groups_norm.items():
        X = norm[cols].to_numpy(float)
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite values in feature group {group_name}.")
        if X.min() < -1e-8 or X.max() > 1 + 1e-8:
            raise ValueError(f"Features in group {group_name} are not normalized to [0, 1].")

    return norm, y, groups, label_order, feature_groups_norm, raw_groups


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
    """Gaussian population encoding.

    response_normalization='max' reproduces the previous tuning script:
        responses are divided by their maximum within each feature.
    response_normalization='raw' uses raw Gaussian responses.
    """
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
                "method": "snnTorch",
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
    return make_result("snnTorch", yt, yp, rows, labels), np.asarray(losses_all, dtype=float)


# ---------------------------------------------------------------------
# Spike-count delta readout
# ---------------------------------------------------------------------

def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def train_delta_readout(S_train, y_train, n_classes, epochs=400, lr=0.05, l2=1e-4, seed=42):
    rng = np.random.default_rng(seed)
    X_counts = S_train.sum(axis=1).astype(float)
    X_counts = X_counts / (X_counts.max(axis=0, keepdims=True) + 1e-8)
    W = rng.normal(0, 0.05, size=(X_counts.shape[1], n_classes))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y_train]
    losses = []
    for _ in range(int(epochs)):
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


def predict_delta_readout(S_test, W, b):
    X_counts = S_test.sum(axis=1).astype(float)
    X_counts = X_counts / (X_counts.max(axis=0, keepdims=True) + 1e-8)
    logits = X_counts @ W + b
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
        W, b, losses = train_delta_readout(
            S_all[tr], y[tr], len(labels),
            epochs=delta_epochs, lr=delta_lr, l2=delta_l2, seed=seed + fold,
        )
        pred, logits = predict_delta_readout(S_all[te], W, b)
        losses_all.append(losses)
        for j, idx in enumerate(te):
            row = {
                "method": "spike_delta",
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
    return make_result("spike_delta", yt, yp, rows, labels), np.asarray(losses_all, dtype=float)


# ---------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------

def run_classical_loso(X, y, groups, labels):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVM": SVC(kernel="linear", class_weight="balanced"),
        "RBFSVM": SVC(kernel="rbf", class_weight="balanced"),
    }
    results = []
    for name, clf in models.items():
        yt, yp, rows = [], [], []
        for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups), start=1):
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            for j, idx in enumerate(te):
                rows.append({
                    "method": name,
                    "fold": int(fold),
                    "subject_id": str(groups[idx]),
                    "y_true": int(y[idx]),
                    "y_pred": int(pred[j]),
                    "true_class": labels[int(y[idx])],
                    "pred_class": labels[int(pred[j])],
                })
            yt.extend(y[te].tolist())
            yp.extend(pred.tolist())
        results.append(make_result(name, yt, yp, rows, labels))
    return results


# ---------------------------------------------------------------------
# Metrics and plotting
# ---------------------------------------------------------------------

def make_result(method, yt, yp, rows, labels):
    cm = confusion_matrix(yt, yp, labels=list(range(len(labels))))
    metrics = {
        "method": method,
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "weighted_f1": float(f1_score(yt, yp, average="weighted")),
        "n_predictions": int(len(yt)),
    }
    return {"metrics": metrics, "predictions": pd.DataFrame(rows), "cm": cm}


def save_confusion_matrix(cm, labels, outpath, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_metric_comparison(metrics_df, outpath):
    plot_df = metrics_df.copy()
    order = ["tonic", "phasic", "psd", "all"]
    plot_df["feature_group"] = pd.Categorical(plot_df["feature_group"], categories=order, ordered=True)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.barplot(data=plot_df, x="feature_group", y="balanced_accuracy", hue="method", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Feature-group comparison across model families")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="feature_group_comparison_outputs")
    parser.add_argument("--task_order", default=None)
    parser.add_argument("--use_existing_norm", action="store_true")
    parser.add_argument("--include_ratio", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=1000)

    # Fixed best snnTorch config from tuning result.
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--max_spikes", type=int, default=10)
    parser.add_argument("--n_pop", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Encoding choice. Default reproduces your best configuration script.
    parser.add_argument("--response_normalization", choices=["max", "raw"], default="max")

    # Spike-delta readout settings.
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

    norm, y, groups, labels, feature_groups, raw_groups = prepare_data(args)
    norm.to_csv(out / "features_normalized_for_feature_group_comparison.csv", index=False)

    snn_cfg = {
        "hidden": args.hidden,
        "beta": args.beta,
        "epochs": args.epochs,
        "lr": args.lr,
        "encoding": "population",
        "T": args.T,
        "max_spikes": args.max_spikes,
        "n_pop": args.n_pop,
        "sigma": args.sigma,
        "weight_decay": args.weight_decay,
    }

    all_metric_rows = []
    all_predictions = []
    summary_results = {}
    t0 = time.time()

    for group_name in ["tonic", "phasic", "psd", "all"]:
        cols = feature_groups[group_name]
        X = norm[cols].to_numpy(float)
        print(f"\nFeature group: {group_name} | features={cols} | n_features={X.shape[1]}")

        group_results = []

        # snnTorch fixed model
        snn_res, snn_losses = run_snn_loso(
            X, y, groups, labels, snn_cfg, args.device,
            response_normalization=args.response_normalization,
            seed=args.seed,
        )
        group_results.append(snn_res)
        pd.DataFrame(snn_losses).to_csv(out / f"losses_snntorch_{group_name}.csv", index=False)

        # spike-delta readout
        delta_res, delta_losses = run_delta_loso(
            X, y, groups, labels, snn_cfg,
            response_normalization=args.response_normalization,
            seed=args.seed,
            delta_epochs=args.delta_epochs,
            delta_lr=args.delta_lr,
            delta_l2=args.delta_l2,
        )
        group_results.append(delta_res)
        pd.DataFrame(delta_losses).to_csv(out / f"losses_spike_delta_{group_name}.csv", index=False)

        # classical baselines
        group_results.extend(run_classical_loso(X, y, groups, labels))

        summary_results[group_name] = {}
        for res in group_results:
            method = res["metrics"]["method"]
            metrics = {**res["metrics"], "feature_group": group_name, "n_features": int(X.shape[1]), "input_channels": int(X.shape[1] * args.n_pop)}
            all_metric_rows.append(metrics)
            summary_results[group_name][method] = metrics

            pred = res["predictions"].copy()
            pred.insert(0, "feature_group", group_name)
            pred.to_csv(pred_dir / f"predictions_{group_name}_{method}.csv", index=False)
            all_predictions.append(pred)

            cm = res["cm"]
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(cm_dir / f"confusion_matrix_{group_name}_{method}.csv")
            save_confusion_matrix(cm, labels, cm_dir / f"confusion_matrix_{group_name}_{method}.png", f"{method} | {group_name}")

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df = metrics_df.sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    metrics_df.to_csv(out / "all_feature_group_model_metrics.csv", index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_csv(out / "all_feature_group_model_predictions.csv", index=False)

    plot_metric_comparison(metrics_df, out / "feature_group_model_comparison_balanced_accuracy.png")

    summary = {
        "input_file": args.input,
        "device_used": args.device,
        "n_rows": int(len(norm)),
        "n_subjects": int(len(np.unique(groups))),
        "classes": labels,
        "raw_feature_groups": raw_groups,
        "normalized_feature_groups": feature_groups,
        "fixed_snntorch_config": snn_cfg,
        "response_normalization": args.response_normalization,
        "spike_delta_config": {
            "epochs": args.delta_epochs,
            "lr": args.delta_lr,
            "l2": args.delta_l2,
            "encoding": "population",
            "T": args.T,
            "max_spikes": args.max_spikes,
            "n_pop": args.n_pop,
            "sigma": args.sigma,
            "response_normalization": args.response_normalization,
        },
        "elapsed_seconds": float(time.time() - t0),
        "results": summary_results,
        "note": "snnTorch uses the fixed best hyperparameters supplied by the user. Classical ML and spike-delta are evaluated with the same LOSO splits for each feature group.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    zip_path = Path(str(out) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            z.write(p, arcname=f"{out.name}/{p.relative_to(out)}")

    print("\nTop results:")
    print(metrics_df.head(20).to_string(index=False))
    print(f"\nOutputs saved to: {out}")
    print(f"Zip saved to: {zip_path}")


if __name__ == "__main__":
    main()

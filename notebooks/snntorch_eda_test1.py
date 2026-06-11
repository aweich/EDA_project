#!/usr/bin/env python3
"""
Tune a real snnTorch spiking neural network for EDA task classification.

Default hyperparameter grid:
    hidden neurons: 16, 32, 64
    beta:           0.85, 0.90, 0.95
    epochs:         50, 100, 200
    learning rate:  0.0005, 0.001, 0.005
    encoding:       Gaussian population encoding only
    max_spikes:     4, 6, 10

Evaluation:
    Leave-one-subject-out cross-validation using Sample, Subject, or subject_id.
    No data augmentation.
    Supports Apple Silicon GPU acceleration via PyTorch MPS using --device mps or --device auto.

Examples:
    python tune_snntorch_eda_29subjects_mps.py --input eda_features.csv --output tuned_snn_outputs --device auto
    python tune_snntorch_eda_29subjects_mps.py --input eda_features_29_samples.csv --output tuned_snn_outputs --task_order "rest0,task1,task2,task3" --device mps
    python tune_snntorch_eda_29subjects_mps.py --input eda_features.csv --max_configs 5 --device mps

Scientific caution:
    This script tunes and evaluates on the same LOSO procedure. This is useful for exploratory model selection.
    For an unbiased final estimate, use nested LOSO or evaluate the selected configuration on an independent cohort.
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


def choose_device(requested: str) -> str:
    """Choose PyTorch device with Apple Silicon MPS support.

    auto priority:
        1. CUDA, for NVIDIA systems
        2. MPS, for Apple Silicon Macs such as M1/M2/M3/M4
        3. CPU fallback
    """
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


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def detect_features(df: pd.DataFrame, use_existing_norm: bool, exclude_mode: bool) -> Tuple[List[str], bool]:
    excluded = {"Sample", "Subject", "subject", "sample", "subject_id", "Task"}
    numeric = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    if use_existing_norm:
        feats = [c for c in numeric if c.endswith("_norm")]
        if exclude_mode:
            feats = [c for c in feats if "mode" not in c.lower()]
        if not feats:
            raise ValueError("--use_existing_norm set, but no numeric *_norm columns found.")
        return feats, True

    raw = [c for c in numeric if not c.endswith("_norm") and c != "EDASymp_Ratio"]
    if exclude_mode:
        raw = [c for c in raw if "mode" not in c.lower()]
    if raw:
        return raw, False

    feats = [c for c in numeric if c.endswith("_norm")]
    if exclude_mode:
        feats = [c for c in feats if "mode" not in c.lower()]
    if not feats:
        raise ValueError("No numeric feature columns found.")
    return feats, True


def prepare_data(args):
    df = read_table(args.input)
    if "Task" not in df.columns:
        raise ValueError("Input file must contain a Task column.")

    subj_col = infer_subject_column(df)
    df["subject_id"] = df[subj_col].astype(str)
    df["Task"] = df["Task"].astype(str)

    task_order = infer_task_order(list(pd.unique(df["Task"])), args.task_order)
    df = df[df["Task"].isin(task_order)].copy()
    df["Task"] = pd.Categorical(df["Task"], categories=task_order, ordered=True)
    df = df.sort_values(["subject_id", "Task"]).reset_index(drop=True)
    df["Task"] = df["Task"].astype(str)

    feature_cols, already_norm = detect_features(df, args.use_existing_norm, args.exclude_mode)
    norm = df.copy()

    if already_norm:
        model_features = feature_cols
    else:
        model_features = []
        for col in feature_cols:
            mn = norm.groupby("subject_id")[col].transform("min")
            mx = norm.groupby("subject_id")[col].transform("max")
            new_col = col + "_norm"
            norm[new_col] = ((norm[col] - mn) / (mx - mn + 1e-8)).clip(0, 1)
            model_features.append(new_col)

    if args.include_ratio and {"Total Power", "EDASymp Power"}.issubset(set(norm.columns)):
        norm["EDASymp_Ratio"] = norm["EDASymp Power"] / (norm["Total Power"] + 1e-12)
        mn = norm.groupby("subject_id")["EDASymp_Ratio"].transform("min")
        mx = norm.groupby("subject_id")["EDASymp_Ratio"].transform("max")
        norm["EDASymp_Ratio_norm"] = ((norm["EDASymp_Ratio"] - mn) / (mx - mn + 1e-8)).clip(0, 1)
        model_features.append("EDASymp_Ratio_norm")

    X = norm[model_features].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Model features contain NaN or inf.")
    if X.min() < -1e-8 or X.max() > 1 + 1e-8:
        raise ValueError("Model features must be normalized to [0, 1]. Use raw columns or *_norm columns.")

    label_to_int = {lab: i for i, lab in enumerate(task_order)}
    y = norm["Task"].map(label_to_int).to_numpy(np.int64)
    groups = norm["subject_id"].to_numpy()
    return norm, X, y, groups, task_order, feature_cols, model_features


def deterministic_rate_encode(X: np.ndarray, T: int, max_spikes: int) -> np.ndarray:
    S = np.zeros((X.shape[0], T, X.shape[1]), dtype=np.float32)
    for i in range(X.shape[0]):
        for f in range(X.shape[1]):
            n = int(np.round(float(X[i, f]) * max_spikes))
            if n > 0:
                S[i, np.linspace(0, T - 1, n).astype(int), f] = 1.0
    return S


def gaussian_population_encode(X: np.ndarray, T: int, max_spikes: int, n_pop: int = 3, sigma: float = 0.25) -> np.ndarray:
    centers = np.linspace(0, 1, n_pop)
    responses = []
    for f in range(X.shape[1]):
        vals = X[:, f:f + 1]
        r = np.exp(-0.5 * ((vals - centers[None, :]) / sigma) ** 2)
        r = r / (r.max(axis=1, keepdims=True) + 1e-8)
        responses.append(r)
    return deterministic_rate_encode(np.concatenate(responses, axis=1), T=T, max_spikes=max_spikes)


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


def train_model(S_train, y_train, n_classes, hidden, beta, epochs, lr, seed, device):
    set_all_seeds(seed)
    X_t = torch.tensor(S_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    model = SpikeSNN(S_train.shape[2], n_classes, hidden, beta).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
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
    return out.argmax(1).cpu().numpy(), out.cpu().numpy()


def run_loso_config(X, y, groups, labels, cfg, device, seed=1000):
    S_all = gaussian_population_encode(X, T=cfg["T"], max_spikes=cfg["max_spikes"], n_pop=cfg["n_pop"], sigma=cfg["sigma"])
    yt, yp, rows, losses_all = [], [], [], []
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, groups), start=1):
        if set(groups[tr]).intersection(set(groups[te])):
            raise RuntimeError("Subject leakage detected.")
        model, losses = train_model(S_all[tr], y[tr], len(labels), cfg["hidden"], cfg["beta"], cfg["epochs"], cfg["lr"], seed + fold, device)
        pred, out = predict_model(model, S_all[te], device)
        losses_all.append(losses)
        for j, idx in enumerate(te):
            row = {
                "fold": fold,
                "subject_id": groups[idx],
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
        "n_predictions": len(yt),
        "n_subjects": len(np.unique(groups)),
    }
    return metrics, pd.DataFrame(rows), cm, np.asarray(losses_all, dtype=float)


def run_classical(X, y, groups):
    rows = []
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "LinearSVM": SVC(kernel="linear", class_weight="balanced"),
        "RBFSVM": SVC(kernel="rbf", class_weight="balanced"),
    }
    for name, clf in models.items():
        yt, yp = [], []
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            yt.extend(y[te].tolist())
            yp.extend(pred.tolist())
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "weighted_f1": float(f1_score(yt, yp, average="weighted")),
        })
    return pd.DataFrame(rows)


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
                        })
    return configs[:args.max_configs] if args.max_configs else configs


def plot_outputs(results, cm, labels, losses, X, best_cfg, outdir):
    top = results.sort_values("balanced_accuracy", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    lab = [f"h={int(r.hidden)}, beta={r.beta}, ep={int(r.epochs)}, lr={r.lr}, sp={int(r.max_spikes)}" for _, r in top.iterrows()]
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["balanced_accuracy"].values, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lab, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced accuracy")
    ax.set_title("Top tuned snnTorch configurations")
    fig.tight_layout()
    fig.savefig(outdir / "top_snntorch_configs.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Best tuned snnTorch confusion matrix")
    fig.tight_layout()
    fig.savefig(outdir / "best_snntorch_confusion_matrix.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses.mean(axis=0))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Mean LOSO training loss, best config")
    fig.tight_layout()
    fig.savefig(outdir / "best_snntorch_training_loss.png", dpi=200)
    plt.close(fig)

    S0 = gaussian_population_encode(X[:1], T=best_cfg["T"], max_spikes=best_cfg["max_spikes"], n_pop=best_cfg["n_pop"], sigma=best_cfg["sigma"])[0]
    ti, fi = np.where(S0.T == 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(ti, fi, marker="|", s=40)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Input channel")
    ax.set_title("Example population-encoded spike raster")
    fig.tight_layout()
    fig.savefig(outdir / "best_snntorch_example_spike_raster.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="tuned_snntorch_outputs")
    parser.add_argument("--task_order", default=None)
    parser.add_argument("--use_existing_norm", action="store_true")
    parser.add_argument("--exclude_mode", action="store_true")
    parser.add_argument("--include_ratio", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use: auto, cpu, cuda, or mps for Apple Silicon GPU.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_configs", type=int, default=None)
    parser.add_argument("--hidden_values", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--beta_values", type=float, nargs="+", default=[0.85, 0.90, 0.95])
    parser.add_argument("--epoch_values", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--lr_values", type=float, nargs="+", default=[0.0005, 0.001, 0.005])
    parser.add_argument("--max_spikes_values", type=int, nargs="+", default=[4, 6, 10])
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n_pop", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.25)
    args = parser.parse_args()

    args.device = choose_device(args.device)
    print(f"Using device: {args.device}")

    out = Path(args.output)
    out.mkdir(exist_ok=True)

    norm, X, y, groups, labels, raw_features, model_features = prepare_data(args)
    norm.to_csv(out / "features_used_for_snntorch_tuning.csv", index=False)

    configs = build_grid(args)
    print(f"Rows={len(norm)}, subjects={len(np.unique(groups))}, classes={labels}")
    print(f"Features ({len(model_features)}): {model_features}")
    print(f"Input channels after population encoding: {X.shape[1] * args.n_pop}")
    print(f"Running {len(configs)} configurations on {args.device}")

    results = []
    best = None
    t0 = time.time()
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg}", flush=True)
        metrics, preds, cm, losses = run_loso_config(X, y, groups, labels, cfg, args.device, args.seed)
        row = {**metrics, "config_id": i, "n_features": X.shape[1], "input_channels": X.shape[1] * cfg["n_pop"]}
        results.append(row)
        if best is None or row["balanced_accuracy"] > best[0]["balanced_accuracy"]:
            best = (row, preds, cm, losses)
        pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False).to_csv(out / "snntorch_tuning_results_partial.csv", index=False)

    results_df = pd.DataFrame(results).sort_values("balanced_accuracy", ascending=False)
    best_row, best_preds, best_cm, best_losses = best
    baselines = run_classical(X, y, groups)

    results_df.to_csv(out / "snntorch_tuning_results.csv", index=False)
    best_preds.to_csv(out / "best_snntorch_predictions.csv", index=False)
    pd.DataFrame(best_cm, index=labels, columns=labels).to_csv(out / "best_snntorch_confusion_matrix.csv")
    baselines.to_csv(out / "classical_baselines_same_features.csv", index=False)

    summary = {
        "input_file": args.input,
        "device_used": args.device,
        "n_rows": int(len(norm)),
        "n_subjects": int(len(np.unique(groups))),
        "subjects": sorted(np.unique(groups).tolist()),
        "classes": labels,
        "raw_feature_cols": raw_features,
        "model_feature_cols": model_features,
        "input_channels_after_population_encoding": int(X.shape[1] * args.n_pop),
        "grid": {
            "hidden_values": args.hidden_values,
            "beta_values": args.beta_values,
            "epoch_values": args.epoch_values,
            "lr_values": args.lr_values,
            "encoding": "population",
            "max_spikes_values": args.max_spikes_values,
            "T": args.T,
            "n_pop": args.n_pop,
            "sigma": args.sigma,
        },
        "n_configs_run": int(len(configs)),
        "elapsed_seconds": float(time.time() - t0),
        "best_snntorch_config": best_row,
        "classical_baselines_same_features": baselines.to_dict(orient="records"),
        "warning": "Exploratory tuning. For unbiased final performance, use nested LOSO or independent held-out data.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    plot_outputs(results_df, best_cm, labels, best_losses, X, best_row, out)

    zip_path = Path(str(out) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.glob("*"):
            z.write(p, arcname=f"{out.name}/{p.name}")

    print("\nTop configurations:")
    print(results_df.head(15).to_string(index=False))
    print("\nClassical baselines:")
    print(baselines.to_string(index=False))
    print("\nBest configuration:")
    print(json.dumps(best_row, indent=2, default=str))
    print(f"\nOutputs saved to {out}")
    print(f"Zip saved to {zip_path}")


if __name__ == "__main__":
    main()

"""Génère les figures et le tableau récapitulatif à partir de results/*.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

ORDER = ["head", "lora_r4", "lora_r8", "lora_r16", "lora_scratch_r8", "full"]
LABELS = {
    "head": "Tête seule",
    "lora_r4": "LoRA r=4",
    "lora_r8": "LoRA r=8",
    "lora_r16": "LoRA r=16",
    "lora_scratch_r8": "LoRA maison r=8",
    "full": "Fine-tuning complet",
}
COLORS = {
    "head": "#9aa3ad",
    "lora_r4": "#7fb8e6",
    "lora_r8": "#2f7fc1",
    "lora_r16": "#1b4f80",
    "lora_scratch_r8": "#3aa37a",
    "full": "#d1603d",
}


def load() -> dict[str, dict]:
    runs = {p.stem: json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")}
    return {k: runs[k] for k in ORDER if k in runs}


def _bar(ax, runs, values, ylabel, fmt="{:.1f}", log=False):
    names = list(runs)
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=[COLORS[n] for n in names])
    ax.set_xticks(x, [LABELS[n] for n in names], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    for b, v in zip(bars, values):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    runs = load()
    if not runs:
        raise SystemExit("Aucun résultat dans results/")

    # 1. Précision et paramètres entraînables
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _bar(axes[0], runs, [100 * r["metrics"]["accuracy"] for r in runs.values()], "Accuracy validation (%)", "{:.2f}")
    accs = [100 * r["metrics"]["accuracy"] for r in runs.values()]
    axes[0].set_ylim(min(accs) - 3, max(accs) + 1.5)
    _bar(axes[1], runs, [r["params"]["trainable"] for r in runs.values()], "Paramètres entraînables (log)", "{:,.0f}", log=True)
    save(fig, "accuracy_params")

    # 2. Profilage : temps et mémoire
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _bar(axes[0], runs, [r["profiling"]["train_time_s"] / 60 for r in runs.values()], "Durée d'entraînement (min)")
    _bar(axes[1], runs, [r["profiling"]["gpu_peak_train_mb"] for r in runs.values()], "Pic mémoire GPU (Mo)", "{:.0f}")
    _bar(axes[2], runs, [r["profiling"]["checkpoint_size_mb"] for r in runs.values()], "Taille sauvegardée (Mo, log)", "{:.2f}", log=True)
    save(fig, "profiling")

    # 3. Courbes de perte
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in runs.items():
        pts = [(h["step"], h["train_loss"]) for h in r["history"] if "train_loss" in h]
        if pts:
            s, l = zip(*pts)
            k = 5
            smooth = np.convolve(l, np.ones(k) / k, mode="valid")
            ax.plot(s[k - 1:], smooth, label=LABELS[name], color=COLORS[name])
    ax.set_xlabel("Itération")
    ax.set_ylabel("Perte d'entraînement (moyenne glissante)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "loss_curves")

    # 4. Compromis précision / paramètres
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in runs.items():
        ax.scatter(r["params"]["trainable"], 100 * r["metrics"]["accuracy"], s=70, color=COLORS[name], label=LABELS[name])
    ax.set_xscale("log")
    ax.set_xlabel("Paramètres entraînables (échelle log)")
    ax.set_ylabel("Accuracy validation (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "tradeoff")

    # 5. Matrice de confusion du meilleur LoRA
    best = max((n for n in runs if n.startswith("lora")), key=lambda n: runs[n]["metrics"]["accuracy"], default=None)
    if best:
        cm = np.array(runs[best]["metrics"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["négatif", "positif"])
        ax.set_yticks([0, 1], ["négatif", "positif"])
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(LABELS[best])
        save(fig, "confusion_matrix")

    # Tableau récapitulatif (LaTeX + texte)
    rows = []
    for name, r in runs.items():
        p, m = r["profiling"], r["metrics"]
        rows.append((LABELS[name], r["params"]["trainable"], r["params"]["ratio_pct"], 100 * m["accuracy"], 100 * m["f1"],
                     p["train_time_s"] / 60, p["train_samples_per_s"], p["gpu_peak_train_mb"], p["checkpoint_size_mb"]))
    tex = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
           "Méthode & Param. entr. & \\% & Acc. & F1 & Durée (min) & Éch./s & GPU (Mo) & Disque (Mo) \\\\", "\\midrule"]
    for r in rows:
        tex.append(f"{r[0]} & {r[1]:,} & {r[2]:.2f} & {r[3]:.2f} & {r[4]:.2f} & {r[5]:.1f} & {r[6]:.0f} & {r[7]:.0f} & {r[8]:.2f} \\\\".replace(",", "\\,"))
        tex[-1] = re.sub(r"(\d)\.(\d)", r"\1{,}\2", tex[-1])  # virgule décimale française
    tex += ["\\bottomrule", "\\end{tabular}"]
    (FIG_DIR / "summary_table.tex").write_text("\n".join(tex))
    for r in rows:
        print(f"{r[0]:<22} {r[1]:>12,} {r[2]:6.2f}% acc={r[3]:.2f} f1={r[4]:.2f} {r[5]:5.1f}min {r[6]:5.0f}éch/s {r[7]:5.0f}Mo {r[8]:8.2f}Mo")


if __name__ == "__main__":
    main()

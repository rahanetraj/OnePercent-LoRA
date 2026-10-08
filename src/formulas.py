"""Rend les formules clés en PNG (mathtext matplotlib) pour la présentation."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "cm"

OUT = Path(__file__).resolve().parent.parent / "figures" / "formulas"
FORMULAS = {
    "lora": r"$h = W_0\,x + \frac{\alpha}{r}\,B\,A\,x,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k}$",
    "init": r"$B = 0 \;\Rightarrow\; \Delta W = BA = 0\ \mathrm{au\ d\acute{e}part}$",
    "params": r"$\rho = \frac{r\,(d+k)}{d\,k} = \frac{8 \cdot 1536}{768^2} \approx 2{,}08\,\%$",
    "merge": r"$W = W_0 + \frac{\alpha}{r}\,B\,A \quad (\mathrm{aucune\ latence\ ajout\acute{e}e})$",
    "grad": r"$\frac{\partial \mathcal{L}}{\partial B} = s\,G\,(Ax)^{\top},\quad \frac{\partial \mathcal{L}}{\partial A} = s\,B^{\top} G\, x^{\top}$",
    "memory": r"$M \approx 4\,(|\Theta| + 3\,|\Theta_{tr}|)\ \mathrm{octets}$",
}

OUT.mkdir(parents=True, exist_ok=True)
for name, tex in FORMULAS.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=26, color="#1d2b36")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)

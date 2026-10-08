"""Application de démonstration : analyse de sentiment avec DistilBERT + adaptateur LoRA.

Lancement :  python app.py   puis ouvrir http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
BASE_MODEL = "distilbert-base-uncased"
LABELS = ["Négatif", "Positif"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cache: dict[str, tuple] = {}


def available_adapters() -> list[str]:
    return sorted(p.name for p in MODELS_DIR.glob("lora_r*") if (p / "adapter_config.json").exists())


def load(adapter: str):
    if adapter not in _cache:
        tok = AutoTokenizer.from_pretrained(MODELS_DIR / adapter)
        base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
        model = PeftModel.from_pretrained(base, MODELS_DIR / adapter)
        model = model.merge_and_unload().to(device).eval()  # W = W0 + (alpha/r) B A
        with torch.no_grad():  # préchauffage CUDA : la latence affichée ensuite est la vraie
            model(**tok("warm up", return_tensors="pt").to(device))
        _cache[adapter] = (tok, model)
    return _cache[adapter]


def adapter_info(adapter: str) -> str:
    path = RESULTS_DIR / f"{adapter}.json"
    if not path.exists():
        return ""
    r = json.loads(path.read_text())
    p, m = r["profiling"], r["metrics"]
    return (
        f"**{adapter}** — paramètres entraînables : {r['params']['trainable']:,} "
        f"({r['params']['ratio_pct']:.2f} %) · accuracy SST-2 : {100 * m['accuracy']:.2f} % · "
        f"F1 : {100 * m['f1']:.2f} % · entraînement : {p['train_time_s'] / 60:.1f} min · "
        f"adaptateur : {p['checkpoint_size_mb']:.2f} Mo"
    )


@torch.no_grad()
def predict(text: str, adapter: str):
    if not text.strip():
        return {}, ""
    tok, model = load(adapter)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    t0 = time.perf_counter()
    probs = model(**inputs).logits.softmax(-1)[0].cpu().tolist()
    ms = (time.perf_counter() - t0) * 1000
    return {LABELS[i]: p for i, p in enumerate(probs)}, f"Latence : {ms:.1f} ms sur {device}"


def build() -> gr.Blocks:
    adapters = available_adapters()
    if not adapters:
        raise SystemExit("Aucun adaptateur dans models/ — lancez d'abord l'entraînement (run_experiments.sh).")
    default = "lora_r8" if "lora_r8" in adapters else adapters[0]

    with gr.Blocks(title="LoRA — Analyse de sentiment") as demo:
        gr.Markdown("# Analyse de sentiment avec LoRA (PEFT)\n"
                    "DistilBERT gelé + adaptateurs de rang faible entraînés sur SST-2. "
                    "Écrivez une phrase en anglais.")
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Texte", lines=4, placeholder="This movie was absolutely wonderful!")
                adapter = gr.Dropdown(adapters, value=default, label="Adaptateur LoRA")
                btn = gr.Button("Analyser", variant="primary")
            with gr.Column(scale=2):
                out = gr.Label(label="Prédiction")
                latency = gr.Markdown()
        info = gr.Markdown(adapter_info(default))
        gr.Examples(
            ["This movie was absolutely wonderful, I loved every minute.",
             "A boring, predictable and painfully long film.",
             "The acting was fine but the story made no sense."],
            inputs=text,
        )
        btn.click(predict, [text, adapter], [out, latency])
        text.submit(predict, [text, adapter], [out, latency])
        adapter.change(adapter_info, adapter, info)
    return demo


if __name__ == "__main__":
    build().launch()

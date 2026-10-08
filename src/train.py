"""Entraînement et profilage : fine-tuning complet vs LoRA (PEFT) vs LoRA « from scratch ».

Exemples :
    python -m src.train --method full
    python -m src.train --method lora --rank 8
    python -m src.train --method lora_scratch --rank 8
    python -m src.train --method head
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from .lora_scratch import inject_lora
from .profiling import Timer, count_parameters, gpu_peak_mb, machine_info, reset_gpu_peak

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
log = logging.getLogger("train")

DEFAULT_LR = {"full": 2e-5, "head": 1e-3, "lora": 5e-4, "lora_scratch": 5e-4}
HEAD_MODULES = ("pre_classifier", "classifier")


@dataclass
class Config:
    method: str = "lora"
    model_name: str = "distilbert-base-uncased"
    rank: int = 8
    alpha: float = 16.0
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q_lin", "v_lin")
    epochs: int = 2
    batch_size: int = 32
    lr: float | None = None
    max_length: int = 128
    train_samples: int = 20000
    seed: int = 42
    run_name: str = ""

    def __post_init__(self) -> None:
        if self.lr is None:
            self.lr = DEFAULT_LR[self.method]
        if not self.run_name:
            self.run_name = self.method + (f"_r{self.rank}" if self.method.startswith("lora") else "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: Config, tokenizer) -> tuple[DataLoader, DataLoader]:
    ds = load_dataset("glue", "sst2")
    train = ds["train"].shuffle(seed=cfg.seed)
    if cfg.train_samples:
        train = train.select(range(min(cfg.train_samples, len(train))))
    val = ds["validation"]

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=cfg.max_length)

    cols = ["sentence", "idx"]
    train = train.map(tok, batched=True, remove_columns=cols).rename_column("label", "labels")
    val = val.map(tok, batched=True, remove_columns=cols).rename_column("label", "labels")
    collator = DataCollatorWithPadding(tokenizer)
    return (
        DataLoader(train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator),
        DataLoader(val, batch_size=64, collate_fn=collator),
    )


def build_model(cfg: Config) -> torch.nn.Module:
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    if cfg.method == "full":
        return model
    if cfg.method == "head":
        for name, p in model.named_parameters():
            p.requires_grad_(any(h in name for h in HEAD_MODULES))
        return model
    if cfg.method == "lora":
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            modules_to_save=list(HEAD_MODULES),
        )
        return get_peft_model(model, peft_cfg)
    if cfg.method == "lora_scratch":
        return inject_lora(model, cfg.target_modules, cfg.rank, cfg.alpha, cfg.lora_dropout, HEAD_MODULES)
    raise ValueError(f"Méthode inconnue : {cfg.method}")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, labels, losses = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.item())
        preds.extend(out.logits.argmax(-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    model.train()
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer = Timer()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    with timer.section("data"):
        train_dl, val_dl = build_dataloaders(cfg, tokenizer)

    model = build_model(cfg).to(device)
    params = count_parameters(model)
    log.info("[%s] paramètres entraînables : %s / %s (%.3f %%)",
             cfg.run_name, f"{params.trainable:,}", f"{params.total:,}", params.ratio)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.01)
    total_steps = cfg.epochs * len(train_dl)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    reset_gpu_peak()
    history: list[dict] = []
    step, n_tokens = 0, 0
    model.train()
    with timer.section("train"):
        for epoch in range(1, cfg.epochs + 1):
            with timer.section(f"epoch_{epoch}"):
                for batch in train_dl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss = model(**batch).loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    n_tokens += int(batch["attention_mask"].sum())
                    if step % 50 == 0:
                        history.append({"step": step, "epoch": epoch, "train_loss": loss.item()})
                    if step % 200 == 0:
                        log.info("[%s] epoch %d step %d/%d loss %.4f", cfg.run_name, epoch, step, total_steps, loss.item())
            metrics = evaluate(model, val_dl, device)
            history.append({"step": step, "epoch": epoch, "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"]})
            log.info("[%s] epoch %d — val acc %.4f f1 %.4f", cfg.run_name, epoch, metrics["accuracy"], metrics["f1"])
    train_peak = gpu_peak_mb()

    # Latence d'inférence (validation complète, 872 phrases)
    reset_gpu_peak()
    with timer.section("inference"):
        final = evaluate(model, val_dl, device)
    infer_peak = gpu_peak_mb()

    # Sauvegarde : adaptateur seul pour LoRA (PEFT), modèle complet sinon
    out_dir = MODELS_DIR / cfg.run_name
    if cfg.method == "lora":
        model.save_pretrained(out_dir)
    elif cfg.method == "lora_scratch":
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v for k, v in model.state_dict().items() if "lora_" in k or any(h in k for h in HEAD_MODULES)},
                   out_dir / "lora_weights.pt")
    else:
        model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    disk_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file() and "token" not in f.name
                  and f.name not in {"vocab.txt", "special_tokens_map.json"}) / 2**20

    train_time = timer.records["train"]
    result = {
        "config": asdict(cfg),
        "params": {"total": params.total, "trainable": params.trainable, "ratio_pct": params.ratio},
        "metrics": final,
        "profiling": {
            "train_time_s": train_time,
            "epoch_times_s": [timer.records[f"epoch_{e}"] for e in range(1, cfg.epochs + 1)],
            "data_prep_s": timer.records["data"],
            "inference_time_s": timer.records["inference"],
            "train_samples_per_s": cfg.epochs * len(train_dl.dataset) / train_time,
            "train_tokens_per_s": n_tokens / train_time,
            "gpu_peak_train_mb": train_peak,
            "gpu_peak_inference_mb": infer_peak,
            "checkpoint_size_mb": disk_mb,
            "steps": step,
        },
        "history": history,
        "machine": machine_info(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{cfg.run_name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("[%s] terminé en %.1f s — acc %.4f — pic GPU %.0f Mo", cfg.run_name, train_time, final["accuracy"], train_peak)
    return result


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=list(DEFAULT_LR), default="lora")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train-samples", type=int, default=20000, help="0 = tout le jeu d'entraînement")
    p.add_argument("--run-name", default="")
    a = p.parse_args()
    return Config(method=a.method, rank=a.rank, alpha=a.alpha, epochs=a.epochs, batch_size=a.batch_size,
                  lr=a.lr, train_samples=a.train_samples, run_name=a.run_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())

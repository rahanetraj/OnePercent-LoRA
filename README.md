# Projet final Python avancé — PEFT / LoRA

**Auteur :** RAHANETRA Fanasina Jason — M1 IA & Big Data
**Enseignant :** RATIARISON Tsinto Aina

Adaptation de DistilBERT à la classification de sentiments (SST-2) avec LoRA, comparée au fine-tuning complet et à l'entraînement de la tête seule. Chaque expérience est profilée : temps, mémoire GPU, débit et taille sur disque.

## Structure

```
src/lora_scratch.py   implémentation de LoRA à la main (LoRALinear, inject_lora, merge)
src/train.py          boucle d'entraînement + profilage (full | head | lora | lora_scratch)
src/profiling.py      chronométrage synchronisé CUDA, mémoire GPU, infos machine
src/plots.py          figures + tableau LaTeX à partir de results/*.json
src/formulas.py       formules rendues en PNG pour la présentation
app.py                application Gradio (démonstration)
tests/                tests unitaires de LoRA (pytest)
results/              métriques et profilage de chaque expérience (JSON)
figures/              graphiques générés
report/rapport.tex    rapport IMRED (LaTeX)
presentation/         présentation de 10 diapositives (.pptx)
```

## Installation

GTX 1050 (Pascal, sm_61) : PyTorch 2.5.1 + CUDA 12.1 est la dernière combinaison qui prend en charge cette carte.

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
```

## Utilisation

```bash
.venv/bin/python -m pytest tests          # tests unitaires
./run_experiments.sh                      # 6 expériences + figures (~40 min sur GTX 1050)
.venv/bin/python -m src.train --method lora --rank 8   # une expérience isolée
.venv/bin/python app.py                   # application sur http://127.0.0.1:7860
```

#!/usr/bin/env bash
# Lance toutes les expériences puis génère les figures.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY -m src.train --method head
$PY -m src.train --method lora --rank 4
$PY -m src.train --method lora --rank 8
$PY -m src.train --method lora --rank 16
$PY -m src.train --method lora_scratch --rank 8
$PY -m src.train --method full
$PY -m src.plots

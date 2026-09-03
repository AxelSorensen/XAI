# 🧠 XAI: Concept Bottleneck Models

Jupyter notebooks exploring explainability on the CUB-200-2011 bird dataset using Concept Bottleneck Models and saliency maps.

## Features

- 🐦 **CUB-200-2011 dataset** — bundled bird image dataset with concept annotations
- 🧩 **Concept Bottleneck Model** — trains models (joint, sequential, independent, standard) that predict human-interpretable concepts before the final class label
- 👁️ **Saliency maps** — `XAI1_saliency.ipynb` visualizes what the model attends to
- 🔬 **Multiple training regimes** — `joint_train.py`, `seq_train.py`, `indep_train.py`, `stand_train.py` compare approaches to bottleneck training

## Installation

```bash
git clone <this repo>
cd XAI
pip install -r requirements.txt  # no requirements.txt is checked in — install torch, torchvision, jupyter, numpy, pandas as needed
```

## Usage

Open any of the notebooks in Jupyter:

```bash
jupyter notebook main.ipynb
```

Training scripts can be run directly, e.g.:

```bash
python joint_train.py
```

Configuration (paths, hyperparameters) lives in `config.py`.

## Built with

- Python, PyTorch (via `ConceptBottleneck-master`)
- Jupyter notebooks

## Status

🚧 Research/experimental — a collection of notebooks and training scripts for coursework or research, not a packaged tool. No `requirements.txt` or `README` existed before this one, so dependencies aren't pinned.

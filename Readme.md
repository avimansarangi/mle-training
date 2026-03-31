Project overview
Installation instructions
Conda environment creation
Package installation
How to run ingest/train/score scripts


✅ Sphinx-generated HTML documentation

Uses NumPy docstring style
Covers public modules, classes, and functions

This repository contains a production‑ready, packageable machine learning workflow that supports data ingestion, model training, and model scoring.
The project follows Python packaging best practices using the src/ layout, includes logging, testing, and CLI‑enabled scripts, and is designed for reproducibility and maintainability.

.
├── README.md
├── env.yaml
├── pyproject.toml
├── src/
│   └── ml_workflow/
│       ├── __init__.py
│       ├── ingest_data.py
│       ├── train.py
│       ├── score.py
│       └── utils/
│           └── logging.py
├── tests/
│   ├── unit/
│   └── functional/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   └── models/
├── logs/
└── docs/

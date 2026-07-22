"""Prode Predictor — a Dixon–Coles goal model for the 2026 FIFA World Cup pool.

The package bundles the data pipeline (:mod:`prode.features`), the fitted model
(:mod:`prode.model`) and the team → confederation map (:mod:`prode.confederations`).
Project-level paths are exposed here so every script resolves ``data/`` and
``predictions/`` the same way, regardless of the working directory.
"""
from pathlib import Path

# Repo root: this file lives at <root>/src/prode/__init__.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
FIGURES_DIR = PROJECT_ROOT / "figures"

__all__ = ["PROJECT_ROOT", "DATA_DIR", "PREDICTIONS_DIR", "FIGURES_DIR"]

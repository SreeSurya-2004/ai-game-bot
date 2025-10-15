from __future__ import annotations

import os
from typing import Optional

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODELS_DIR = os.path.abspath(MODELS_DIR)


def ensure_models_dir() -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return MODELS_DIR


def model_path(game_name: str) -> str:
    ensure_models_dir()
    filename = f"{game_name}_qtable.json"
    return os.path.join(MODELS_DIR, filename)


def clear_model(game_name: str) -> None:
    path = model_path(game_name)
    if os.path.exists(path):
        os.remove(path)

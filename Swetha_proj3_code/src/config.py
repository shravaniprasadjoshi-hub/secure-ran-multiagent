from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "telemetry_csv", "rag_corpus", "outputs_dir", "models_dir", "plots_dir", "metrics_dir", "reports_dir", "twin_dir"):
        if key in cfg.get("paths", {}):
            cfg["paths"][key] = str(ROOT / cfg["paths"][key])
    return cfg

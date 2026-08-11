from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_telemetry(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["energy_label"] = df.apply(
        lambda r: "deep_sleep" if r["cell_load_pct"] < 30 else ("mimo_reduce" if r["cell_load_pct"] < 60 else "normal"),
        axis=1,
    )
    df["threat_label"] = df["threat_type"].apply(lambda x: "benign" if x == "none" else "malicious")
    return df


def load_rag_corpus(path: str | Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

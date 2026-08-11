from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_features(df: pd.DataFrame, feature_cols: list[str], fit: bool = True, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    X = df[feature_cols].astype(float).fillna(0).values
    if fit or scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    return X, scaler


def prepare_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    strat = df["scenario_type"]
    train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, stratify=strat, random_state=seed)
    val_size = val_ratio / (val_ratio + (1 - train_ratio - val_ratio))
    val_df, test_df = train_test_split(temp_df, test_size=1 - val_size, stratify=temp_df["scenario_type"], random_state=seed)
    return {"train": train_df.reset_index(drop=True), "val": val_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


def encode_target(y: pd.Series, le: LabelEncoder | None = None, fit: bool = True) -> tuple[np.ndarray, LabelEncoder]:
    if fit or le is None:
        le = LabelEncoder()
        return le.fit_transform(y.astype(str)), le
    return le.transform(y.astype(str)), le

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder

from src.data.preprocessor import encode_features, encode_target


@dataclass
class AgentResult:
    agent_name: str
    task: str
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    model_type: str = ""


def build_model(model_type: str, task: str):
    if task == "classification":
        if model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
        if model_type == "mlp":
            return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42)
        return RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1)
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    if model_type == "mlp":
        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42)
    return GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)


def _cls_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except ValueError:
            pass
    return m


def _reg_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_agent(
    agent_name: str,
    agent_cfg: dict,
    splits: dict,
) -> tuple[Any, dict, AgentResult]:
    features = agent_cfg["features"]
    target = agent_cfg["target"]
    task = agent_cfg["task"]
    model_type = agent_cfg["model"]

    X_train, scaler = encode_features(splits["train"], features, fit=True)
    X_val, _ = encode_features(splits["val"], features, fit=False, scaler=scaler)
    X_test, _ = encode_features(splits["test"], features, fit=False, scaler=scaler)

    model = build_model(model_type, task)
    label_encoder = None

    if task == "classification":
        y_train, label_encoder = encode_target(splits["train"][target], fit=True)
        y_val, _ = encode_target(splits["val"][target], le=label_encoder, fit=False)
        y_test, _ = encode_target(splits["test"][target], le=label_encoder, fit=False)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        result = AgentResult(
            agent_name=agent_name,
            task=task,
            model_type=model_type,
            train_metrics=_cls_metrics(y_train, train_pred),
            val_metrics=_cls_metrics(y_val, val_pred),
            test_metrics=_cls_metrics(y_test, test_pred, proba),
            predictions={"y_test": y_test, "y_pred": test_pred, "y_proba": proba, "classes": label_encoder.classes_.tolist()},
        )
    else:
        y_train = splits["train"][target].astype(float).values
        y_val = splits["val"][target].astype(float).values
        y_test = splits["test"][target].astype(float).values
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        result = AgentResult(
            agent_name=agent_name,
            task=task,
            model_type=model_type,
            train_metrics=_reg_metrics(y_train, train_pred),
            val_metrics=_reg_metrics(y_val, val_pred),
            test_metrics=_reg_metrics(y_test, test_pred),
            predictions={"y_test": y_test, "y_pred": test_pred},
        )

    artifact = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "target": target,
        "task": task,
        "label_encoder": label_encoder,
        "model_type": model_type,
    }
    return artifact, artifact, result

"""LightGBM multiclass model of ``high - anchor`` with temperature-scaling calibration.

The classifier emits a probability for each whole-degree residual in
``[MIN_RESIDUAL_DEGREES, MAX_RESIDUAL_DEGREES]``; adding the anchor turns that into a PMF over
the day's final high in °F, from which any Kalshi contract probability is a partial sum.

Splits are strictly chronological by climate day: train → calibrate (also LightGBM's
early-stopping set) → test. The calibration step fits a single softmax temperature by
minimising log-loss on the calibration split, which is cheap and cannot reorder classes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from weatherbot.config import (
    MAX_RESIDUAL_DEGREES,
    MIN_RESIDUAL_DEGREES,
    models_dir,
)
from weatherbot.features import FEATURE_COLUMNS, load_snapshots

log = logging.getLogger(__name__)

N_CLASSES = MAX_RESIDUAL_DEGREES - MIN_RESIDUAL_DEGREES + 1
RESIDUALS = np.arange(MIN_RESIDUAL_DEGREES, MAX_RESIDUAL_DEGREES + 1)

LGB_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_bin": 255,
    "verbose": -1,
    "num_threads": 0,
    "seed": 7,
}
MAX_ROUNDS = 3000
EARLY_STOPPING = 100


@dataclass
class Split:
    train: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame


def split_by_day(table: pd.DataFrame, train_end: date, calib_end: date) -> Split:
    day = pd.to_datetime(table["day"]).dt.date
    return Split(
        train=table[day <= train_end],
        calib=table[(day > train_end) & (day <= calib_end)],
        test=table[day > calib_end],
    )


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return (frame["target"].to_numpy() - MIN_RESIDUAL_DEGREES).astype(int)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return probs


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Softmax temperature minimising calibration-set NLL (``T > 1`` softens, ``< 1`` sharpens)."""
    res = minimize_scalar(
        lambda t: _nll(_softmax(logits / t), y), bounds=(0.25, 4.0), method="bounded"
    )
    return float(res.x)


class HighModel:
    """Trained booster + calibration temperature, with save/load."""

    def __init__(
        self, booster: lgb.Booster, temperature: float, features: list[str], meta: dict[str, Any]
    ):
        self.booster = booster
        self.temperature = temperature
        self.features = features
        self.meta = meta

    @property
    def train_days(self) -> tuple[str, str]:
        first, last = self.meta["train_days"]
        return str(first), str(last)

    @property
    def calib_days(self) -> tuple[str, str]:
        first, last = self.meta["calib_days"]
        return str(first), str(last)

    # -- inference --

    def raw_logits(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[self.features]
        logits: np.ndarray = np.asarray(self.booster.predict(X, raw_score=True), dtype=float)
        return logits

    def predict_residual_pmf(self, frame: pd.DataFrame) -> np.ndarray:
        """``(n, N_CLASSES)`` calibrated probabilities over :data:`RESIDUALS`."""
        return _softmax(self.raw_logits(frame) / self.temperature)

    def predict_high_pmf(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Support ``(n, N_CLASSES)`` of integer highs (°F) and matching probabilities."""
        pmf = self.predict_residual_pmf(frame)
        support = frame["anchor"].to_numpy()[:, None] + RESIDUALS[None, :]
        return support, pmf

    # -- persistence --

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or models_dir()
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "lgbm.txt"))
        (directory / "model.json").write_text(
            json.dumps(
                {"temperature": self.temperature, "features": self.features, **self.meta},
                indent=2,
                default=str,
            )
        )
        return directory

    @classmethod
    def load(cls, directory: Path | None = None) -> HighModel:
        directory = directory or models_dir()
        meta = json.loads((directory / "model.json").read_text())
        booster = lgb.Booster(model_file=str(directory / "lgbm.txt"))
        return cls(booster, meta.pop("temperature"), meta.pop("features"), meta)


def train(split: Split, features: list[str] | None = None) -> HighModel:
    features = features or list(FEATURE_COLUMNS)
    dtrain = lgb.Dataset(
        split.train[features], _labels(split.train), categorical_feature=["station_id"]
    )
    dcalib = lgb.Dataset(split.calib[features], _labels(split.calib), reference=dtrain)
    log.info("training on %d rows, early-stopping on %d", len(split.train), len(split.calib))
    booster = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dcalib],
        valid_names=["calib"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(200)],
    )
    logits = np.asarray(booster.predict(split.calib[features], raw_score=True), dtype=float)
    y = _labels(split.calib)
    temperature = fit_temperature(logits, y)
    log.info(
        "best_iteration=%d  calib NLL raw=%.4f  scaled(T=%.3f)=%.4f",
        booster.best_iteration,
        _nll(_softmax(logits), y),
        temperature,
        _nll(_softmax(logits / temperature), y),
    )
    meta: dict[str, Any] = {
        "best_iteration": booster.best_iteration,
        "train_rows": len(split.train),
        "calib_rows": len(split.calib),
        "train_days": [str(split.train["day"].min()), str(split.train["day"].max())],
        "calib_days": [str(split.calib["day"].min()), str(split.calib["day"].max())],
        "residual_range": [MIN_RESIDUAL_DEGREES, MAX_RESIDUAL_DEGREES],
        "params": LGB_PARAMS,
    }
    return HighModel(booster, temperature, features, meta)


def train_and_save(train_end: date, calib_end: date) -> HighModel:
    table = load_snapshots()
    split = split_by_day(table, train_end, calib_end)
    model = train(split)
    path = model.save()
    log.info("saved model to %s", path)
    return model

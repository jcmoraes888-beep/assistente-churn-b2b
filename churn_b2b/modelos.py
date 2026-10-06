"""Modelos comparados: Regressão Logística (baseline), LightGBM e Rede Neural (PyTorch).

Todos expõem a mesma interface: fit(X_df, y) / predict_proba(X_df) -> probabilidade de churn.
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from .features import CATEGORICAS, NUMERICAS

warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def preprocessador() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
    ], verbose_feature_names_out=False)


class _Rede(nn.Module):
    def __init__(self, entradas: int, camadas: tuple[int, ...], dropout: float):
        super().__init__()
        blocos, anterior = [], entradas
        for n in camadas:
            blocos += [nn.Linear(anterior, n), nn.BatchNorm1d(n), nn.ReLU(), nn.Dropout(dropout)]
            anterior = n
        blocos.append(nn.Linear(anterior, 1))
        self.rede = nn.Sequential(*blocos)

    def forward(self, x):
        return self.rede(x).squeeze(-1)


class RedeNeuralChurn:
    """MLP para dados tabulares com early stopping pela PR-AUC de validação."""

    def __init__(self, camadas=(64, 32), dropout=0.2, lr=1e-3, weight_decay=1e-4,
                 batch=256, max_epocas=150, paciencia=12, seed=42):
        self.params = dict(camadas=camadas, dropout=dropout, lr=lr, weight_decay=weight_decay,
                           batch=batch, max_epocas=max_epocas, paciencia=paciencia, seed=seed)
        self.historico: list[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        p = self.params
        torch.manual_seed(p["seed"])
        self.modelo = _Rede(X.shape[1], tuple(p["camadas"]), p["dropout"])
        peso_pos = torch.tensor((1 - y.mean()) / y.mean(), dtype=torch.float32)
        perda = nn.BCEWithLogitsLoss(pos_weight=peso_pos)
        otim = torch.optim.AdamW(self.modelo.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        melhor, melhor_estado, sem_melhora = -1.0, None, 0
        for epoca in range(p["max_epocas"]):
            self.modelo.train()
            ordem = torch.randperm(len(Xt))
            perda_total = 0.0
            for ini in range(0, len(Xt), p["batch"]):
                idx = ordem[ini:ini + p["batch"]]
                if len(idx) < 2:
                    continue
                otim.zero_grad()
                l = perda(self.modelo(Xt[idx]), yt[idx])
                l.backward()
                otim.step()
                perda_total += l.item() * len(idx)
            pr_auc = average_precision_score(y_val, self._prever(X_val))
            self.historico.append({"epoca": epoca, "loss": perda_total / len(Xt), "val_pr_auc": pr_auc})
            if pr_auc > melhor + 1e-4:
                melhor, melhor_estado, sem_melhora = pr_auc, copy.deepcopy(self.modelo.state_dict()), 0
            else:
                sem_melhora += 1
                if sem_melhora >= p["paciencia"]:
                    break
        self.modelo.load_state_dict(melhor_estado)
        self.melhor_epoca = int(np.argmax([h["val_pr_auc"] for h in self.historico]))
        return self

    def _prever(self, X: np.ndarray) -> np.ndarray:
        self.modelo.eval()
        with torch.no_grad():
            return torch.sigmoid(self.modelo(torch.tensor(X, dtype=torch.float32))).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._prever(X)
        return np.column_stack([1 - p, p])


class ModeloChurn:
    """Pré-processamento + estimador. É o objeto salvo em models/ e usado pela API."""

    def __init__(self, nome: str, estimador):
        self.nome, self.estimador, self.prep = nome, estimador, preprocessador()
        self.calibrador = None

    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame | None = None, y_val=None):
        Xp = self.prep.fit_transform(X)
        if isinstance(self.estimador, RedeNeuralChurn):
            self.estimador.fit(Xp, np.asarray(y), self.prep.transform(X_val), np.asarray(y_val))
        elif isinstance(self.estimador, LGBMClassifier) and X_val is not None:
            from lightgbm import early_stopping, log_evaluation
            self.estimador.fit(Xp, y, eval_set=[(self.prep.transform(X_val), y_val)],
                               eval_metric="average_precision",
                               callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
        else:
            self.estimador.fit(Xp, y)
        return self

    def transformar(self, X: pd.DataFrame) -> np.ndarray:
        return self.prep.transform(X)

    def proba_bruta(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimador.predict_proba(self.prep.transform(X))[:, 1]

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def calibrar(self, X_val: pd.DataFrame, y_val):
        """Platt scaling na validação: o modelo treina com classes balanceadas, o que infla as
        probabilidades. Calibrar faz "risco 30%" significar ~30% de chance real."""
        self.calibrador = LogisticRegression(C=1e6).fit(self._logit(self.proba_bruta(X_val)), y_val)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.proba_bruta(X)
        return self.calibrador.predict_proba(self._logit(p))[:, 1] if self.calibrador is not None else p

    @property
    def nomes_transformados(self) -> list[str]:
        return list(self.prep.get_feature_names_out())


def candidatos() -> dict[str, list[tuple[dict, object]]]:
    """Modelos e grades de hiperparâmetros (pequenas, para rodar em minutos num notebook)."""
    grade_rede = [dict(camadas=c, dropout=d, lr=lr)
                  for c in [(64, 32), (128, 64, 32)] for d in [0.1, 0.3] for lr in [1e-3, 3e-4]]
    grade_lgbm = [dict(num_leaves=nl, learning_rate=lr, min_child_samples=mcs)
                  for nl in [15, 31] for lr in [0.03, 0.08] for mcs in [20, 60]]
    return {
        "regressao_logistica": [(dict(C=c), LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
                                for c in [0.1, 1.0]],
        "lightgbm": [(g, LGBMClassifier(n_estimators=1000, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, is_unbalance=True, random_state=42,
                                        verbose=-1, **g)) for g in grade_lgbm],
        "rede_neural": [(g, RedeNeuralChurn(**g)) for g in grade_rede],
    }

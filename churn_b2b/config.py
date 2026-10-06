"""Configurações centrais do projeto."""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data"
MODELOS = RAIZ / "models"
RELATORIOS = RAIZ / "reports"

SEED = 42
N_CLIENTES = 2500
INICIO = pd.Timestamp("2024-04-01")   # início da simulação
MESES = 30                            # até 30/09/2026

JANELA_CHURN_DIAS = 90   # cliente "churnou" se não comprar nos 90 dias após o corte
JANELA_ATIVO_DIAS = 90   # só pontuamos clientes que compraram nos 90 dias antes do corte

# Cortes (meses desde INICIO) usados para montar os conjuntos — divisão TEMPORAL, sem vazamento
CORTES_TREINO = list(range(9, 22))   # vários retratos empilhados
CORTES_VALIDACAO = [23, 24]
CORTES_TESTE = [26, 27]
CORTE_PRODUCAO = MESES               # retrato atual, sem rótulo (clientes a pontuar)

MLFLOW_URI = f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENTO = "churn-b2b"


def data_corte(mes: int) -> pd.Timestamp:
    return INICIO + pd.DateOffset(months=mes)

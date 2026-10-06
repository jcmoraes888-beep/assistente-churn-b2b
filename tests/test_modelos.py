import numpy as np
import pandas as pd

from churn_b2b.features import ATRIBUTOS, construir_varios
from churn_b2b.modelos import ModeloChurn, RedeNeuralChurn
from churn_b2b.explicar import motivos, valores_shap
from churn_b2b.simulacao import gerar


def test_rede_treina_e_explica():
    t = gerar(n_clientes=300, seed=3, salvar=False)
    cortes = [pd.Timestamp(f"2025-{m:02d}-01") for m in (3, 5, 7)]
    df = construir_varios(t, cortes)
    treino, val = df[df.data_corte < cortes[-1]], df[df.data_corte == cortes[-1]]
    m = ModeloChurn("rede_neural", RedeNeuralChurn(max_epocas=5, paciencia=2))
    m.fit(treino[ATRIBUTOS], treino.churn, val[ATRIBUTOS], val.churn).calibrar(val[ATRIBUTOS], val.churn)
    p = m.predict_proba(val[ATRIBUTOS])
    assert p.shape == (len(val),) and np.all((p >= 0) & (p <= 1))

    amostra = val.head(5)
    sv = valores_shap(m, amostra[ATRIBUTOS], treino[ATRIBUTOS])
    assert list(sv.columns) == ATRIBUTOS
    mot = motivos(amostra, sv, minimo=0.0)
    assert "temas" in mot.columns

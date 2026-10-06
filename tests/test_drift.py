import numpy as np
import pandas as pd

from churn_b2b.monitoramento.drift import drift_por_segmento, psi_categorico, psi_numerico


def test_psi_estavel_e_com_drift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numerico(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.05
    assert psi_numerico(ref, pd.Series(rng.normal(1, 1, 5000))) > 0.25


def test_psi_categorico():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi_categorico(ref, ref) < 1e-9
    assert psi_categorico(ref, pd.Series(["a"] * 90 + ["b"] * 10)) > 0.25


def test_drift_localizado_aparece_por_segmento():
    """Drift em só uma cidade fica diluído no global, mas aparece no recorte por cidade."""
    rng = np.random.default_rng(1)
    n = 3000
    ref = pd.DataFrame({"cidade": ["A"] * (n - 300) + ["B"] * 300, "x": rng.beta(1, 9, n)})
    atual = ref.copy()
    atual.loc[atual.cidade == "B", "x"] = rng.beta(4, 6, 300)
    assert psi_numerico(ref.x, atual.x) < 0.25
    seg = drift_por_segmento(ref, atual, ["x"])
    assert seg.iloc[0].cidade == "B" and seg.iloc[0].psi > 0.25
    assert seg.set_index("cidade").loc["A", "psi"] < 0.1

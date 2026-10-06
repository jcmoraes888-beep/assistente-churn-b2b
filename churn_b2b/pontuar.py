"""Pontua a carteira ATUAL: risco de churn, faixa e os 3 principais motivos de cada cliente.

Uso:  python -m churn_b2b.pontuar
Saída: data/scores_atuais.csv  (entrada do RAG + LLM nas próximas semanas)
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from . import config as C
from .explicar import avaliar_contra_gabarito, motivos, valores_shap
from .features import ATRIBUTOS, construir_retrato
from .simulacao import carregar


def carregar_modelo():
    modelo = joblib.load(C.MODELOS / "modelo_churn.joblib")
    meta = json.loads((C.MODELOS / "metadata.json").read_text("utf-8"))
    return modelo, meta


def faixa(p: float, faixas: dict) -> str:
    return "Alto" if p >= faixas["alto"] else ("Médio" if p >= faixas["medio"] else "Baixo")


def pontuar(corte: pd.Timestamp | None = None, explicar_faixas=("Alto", "Médio")) -> pd.DataFrame:
    t = carregar()
    modelo, meta = carregar_modelo()
    corte = corte or C.data_corte(C.CORTE_PRODUCAO)
    X = construir_retrato(t["clientes"], t["pedidos"], t["atendimentos"], corte, com_rotulo=False)

    X["prob_churn"] = modelo.predict_proba(X[ATRIBUTOS])
    X["faixa_risco"] = X["prob_churn"].apply(faixa, faixas=meta["faixas_risco"])
    alvo = X[X["faixa_risco"].isin(explicar_faixas)]
    fundo = pd.read_csv(C.DADOS / "referencia_teste.csv")[ATRIBUTOS]
    mot = motivos(alvo, valores_shap(modelo, alvo[ATRIBUTOS], fundo))
    X = X.join(mot)
    X = X.merge(t["clientes"][["cliente_id", "razao_social"]], on="cliente_id")
    colunas = ["cliente_id", "razao_social", "segmento", "cidade", "canal", "vendedor", "receita_90d",
               "recencia_dias", "prob_churn", "faixa_risco", "motivo_1", "motivo_2", "motivo_3",
               "temas", "tema_principal"]
    return X[colunas].sort_values("prob_churn", ascending=False).reset_index(drop=True)


def main():
    scores = pontuar()
    scores.to_csv(C.DADOS / "scores_atuais.csv", index=False)
    resumo = scores.faixa_risco.value_counts()
    receita_risco = scores.loc[scores.faixa_risco == "Alto", "receita_90d"].sum()
    print(f"Carteira ativa: {len(scores):,} clientes · Alto: {resumo.get('Alto', 0)} · "
          f"Médio: {resumo.get('Médio', 0)} · Baixo: {resumo.get('Baixo', 0)}")
    print(f"Faturamento trimestral em risco alto: R$ {receita_risco:,.0f}".replace(",", "."))
    print(scores.head(5)[["cliente_id", "segmento", "prob_churn", "motivo_1", "motivo_2"]].to_string(index=False))

    # checagem das explicações contra o gabarito da simulação (no conjunto de teste)
    t = carregar()
    modelo, meta = carregar_modelo()
    teste = pd.read_csv(C.DADOS / "referencia_teste.csv")
    churn = teste[teste.churn == 1].copy()
    churn["prob_churn"] = modelo.predict_proba(churn[ATRIBUTOS])
    churn = churn[churn["prob_churn"] >= meta["faixas_risco"]["medio"]].reset_index(drop=True)
    mot = motivos(churn, valores_shap(modelo, churn[ATRIBUTOS], teste[ATRIBUTOS]))
    aval = avaliar_contra_gabarito(churn, mot, t["verdade"])
    if aval:
        print(f"\nExplicações x causa real (teste): em {aval['causa_real_entre_os_motivos']:.0%} de "
              f"{aval['clientes_avaliados']} churns com causa conhecida, a causa aparece entre os 3 motivos.")
        meta["avaliacao_explicacoes"] = aval
        (C.MODELOS / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()

"""Engenharia de atributos: transforma o histórico em um "retrato" do cliente numa data de corte.

Regra de ouro (sem vazamento): os atributos usam SÓ dados anteriores ao corte.
O rótulo usa SÓ os 90 dias seguintes ao corte.

Definição de churn B2B (não há contrato para cancelar):
    cliente ativo no corte (comprou nos últimos 90 dias) que NÃO compra nos 90 dias seguintes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import JANELA_ATIVO_DIAS, JANELA_CHURN_DIAS

CATEGORICAS = ["segmento", "cidade", "canal", "vendedor"]
NUMERICAS = [
    "recencia_dias", "pedidos_90d", "pedidos_90d_anterior", "tendencia_pedidos",
    "receita_90d", "tendencia_receita", "ticket_medio_90d", "itens_por_pedido_90d", "tendencia_mix",
    "intervalo_medio_dias", "irregularidade_compras", "pct_entregas_atrasadas_90d",
    "atraso_medio_entrega_dias", "desconto_medio_90d", "variacao_desconto",
    "atraso_medio_pagamento_dias", "reclamacoes_entrega_90d", "reclamacoes_preco_90d",
    "reclamacoes_financeiro_90d", "reclamacoes_produto_90d", "tempo_relacionamento_meses",
    "prazo_pagamento_dias",
]
ATRIBUTOS = NUMERICAS + CATEGORICAS

TIPO_PARA_COLUNA = {
    "Atraso na entrega": "reclamacoes_entrega_90d",
    "Preço/condição comercial": "reclamacoes_preco_90d",
    "Cobrança/financeiro": "reclamacoes_financeiro_90d",
    "Qualidade do produto": "reclamacoes_produto_90d",
    "Falta de produto": "reclamacoes_produto_90d",
}


def _razao(atual, anterior):
    """Variação relativa suavizada: -1 (zerou) ... 0 (estável) ... positivo (cresceu)."""
    return (atual - anterior) / (anterior + 1e-9) if np.isscalar(anterior) else \
        np.where(anterior > 0, (atual - anterior) / anterior.clip(lower=1e-9), 0.0)


def construir_retrato(clientes: pd.DataFrame, pedidos: pd.DataFrame, atendimentos: pd.DataFrame,
                      corte: pd.Timestamp, com_rotulo: bool = True) -> pd.DataFrame:
    corte = pd.Timestamp(corte)
    d90, d180 = corte - pd.Timedelta(days=JANELA_ATIVO_DIAS), corte - pd.Timedelta(days=2 * JANELA_ATIVO_DIAS)

    passado = pedidos[pedidos["data"] < corte]
    janela = passado[passado["data"] >= d90]
    anterior = passado[(passado["data"] >= d180) & (passado["data"] < d90)]

    ativos = clientes[clientes["inicio_relacionamento"] < corte]
    ativos = ativos[ativos["cliente_id"].isin(janela["cliente_id"].unique())]
    r = ativos.set_index("cliente_id")[CATEGORICAS + ["prazo_pagamento_dias", "inicio_relacionamento"]].copy()

    g = janela.groupby("cliente_id")
    ga = anterior.groupby("cliente_id")
    r["recencia_dias"] = (corte - passado.groupby("cliente_id")["data"].max()).dt.days
    r["pedidos_90d"] = g.size()
    r["pedidos_90d_anterior"] = ga.size()
    r["receita_90d"] = g["valor"].sum()
    receita_ant = ga["valor"].sum()
    r["ticket_medio_90d"] = g["valor"].mean()
    r["itens_por_pedido_90d"] = g["itens"].mean()
    itens_ant = ga["itens"].mean()
    r["pct_entregas_atrasadas_90d"] = g["atraso_entrega_dias"].apply(lambda s: (s > 0).mean())
    r["atraso_medio_entrega_dias"] = g["atraso_entrega_dias"].mean()
    r["desconto_medio_90d"] = g["desconto_pct"].mean()
    desc_ant = ga["desconto_pct"].mean()
    r["atraso_medio_pagamento_dias"] = g["atraso_pagamento_dias"].mean()

    r = r.fillna({"pedidos_90d_anterior": 0})
    r["tendencia_pedidos"] = _razao(r["pedidos_90d"], r["pedidos_90d_anterior"])
    r["tendencia_receita"] = _razao(r["receita_90d"], receita_ant.reindex(r.index).fillna(0))
    r["tendencia_mix"] = _razao(r["itens_por_pedido_90d"], itens_ant.reindex(r.index).fillna(0))
    r["variacao_desconto"] = (r["desconto_medio_90d"] - desc_ant.reindex(r.index)).fillna(0)

    # regularidade das compras nos últimos 180 dias
    ult180 = passado[passado["data"] >= d180].sort_values("data")
    dif = ult180.groupby("cliente_id")["data"].diff().dt.days
    estat = dif.groupby(ult180["cliente_id"]).agg(["mean", "std"])
    r["intervalo_medio_dias"] = estat["mean"]
    r["irregularidade_compras"] = (estat["std"] / estat["mean"]).replace([np.inf], np.nan)
    r["intervalo_medio_dias"] = r["intervalo_medio_dias"].fillna(JANELA_ATIVO_DIAS)
    r["irregularidade_compras"] = r["irregularidade_compras"].fillna(0)

    at = atendimentos[(atendimentos["data"] < corte) & (atendimentos["data"] >= d90)]
    at = at.assign(col=at["tipo"].map(TIPO_PARA_COLUNA)).groupby(["cliente_id", "col"]).size().unstack()
    for col in set(TIPO_PARA_COLUNA.values()):
        r[col] = at[col].reindex(r.index) if col in at else 0
    r[list(set(TIPO_PARA_COLUNA.values()))] = r[list(set(TIPO_PARA_COLUNA.values()))].fillna(0)

    r["tempo_relacionamento_meses"] = (corte - r.pop("inicio_relacionamento")).dt.days / 30.44
    r = r.reset_index()
    r.insert(1, "data_corte", corte)

    if com_rotulo:
        futuro = pedidos[(pedidos["data"] >= corte) &
                         (pedidos["data"] < corte + pd.Timedelta(days=JANELA_CHURN_DIAS))]
        r["churn"] = (~r["cliente_id"].isin(futuro["cliente_id"].unique())).astype(int)
    return r


def construir_varios(tabelas: dict, cortes: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.concat([construir_retrato(tabelas["clientes"], tabelas["pedidos"], tabelas["atendimentos"], c)
                      for c in cortes], ignore_index=True)

"""Explicações por cliente com SHAP: POR QUE o modelo acha que este cliente vai parar de comprar.

Os motivos viram texto em português e recebem um "tema" (entrega, preço, financeiro, engajamento...),
que na Semana 2 será usado para buscar a estratégia de retenção certa no RAG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from .features import ATRIBUTOS, CATEGORICAS, NUMERICAS
from .modelos import ModeloChurn

# atributo: (tema, como escrever o motivo)
DESCRICOES = {
    "recencia_dias": ("engajamento", lambda v: f"Última compra há {v:.0f} dias"),
    "pedidos_90d": ("engajamento", lambda v: f"Só {v:.0f} pedido(s) nos últimos 90 dias"),
    "pedidos_90d_anterior": ("engajamento", lambda v: f"{v:.0f} pedidos no trimestre anterior"),
    "tendencia_pedidos": ("engajamento", lambda v: f"Pedidos {v:+.0%} em relação ao trimestre anterior"),
    "receita_90d": ("engajamento", lambda v: f"Faturamento de R$ {v:,.0f} nos últimos 90 dias".replace(",", ".")),
    "tendencia_receita": ("engajamento", lambda v: f"Faturamento {v:+.0%} em relação ao trimestre anterior"),
    "ticket_medio_90d": ("engajamento", lambda v: f"Ticket médio de R$ {v:,.0f}".replace(",", ".")),
    "itens_por_pedido_90d": ("produto", lambda v: f"Média de {v:.1f} itens por pedido"),
    "tendencia_mix": ("produto", lambda v: f"Mix de itens por pedido {v:+.0%}"),
    "intervalo_medio_dias": ("engajamento", lambda v: f"Intervalo médio de {v:.0f} dias entre compras"),
    "irregularidade_compras": ("engajamento", lambda v: f"Compras irregulares (variação de {v:.0%} no intervalo)"),
    "pct_entregas_atrasadas_90d": ("entrega", lambda v: f"{v:.0%} das entregas com atraso nos últimos 90 dias"),
    "atraso_medio_entrega_dias": ("entrega", lambda v: f"Atraso médio de entrega de {v:.1f} dia(s)"),
    "desconto_medio_90d": ("preco", lambda v: f"Desconto médio de {v:.1f}%"),
    "variacao_desconto": ("preco", lambda v: f"Desconto {v:+.1f} p.p. em relação ao trimestre anterior"),
    "atraso_medio_pagamento_dias": ("financeiro", lambda v: f"Atraso médio de pagamento de {v:.0f} dias"),
    "reclamacoes_entrega_90d": ("entrega", lambda v: f"{v:.0f} reclamação(ões) de atraso na entrega"),
    "reclamacoes_preco_90d": ("preco", lambda v: f"{v:.0f} reclamação(ões) de preço/condição comercial"),
    "reclamacoes_financeiro_90d": ("financeiro", lambda v: f"{v:.0f} chamado(s) de cobrança/financeiro"),
    "reclamacoes_produto_90d": ("produto", lambda v: f"{v:.0f} reclamação(ões) de produto/falta de item"),
    "tempo_relacionamento_meses": ("relacionamento", lambda v: f"Cliente há {v:.0f} meses"),
    "prazo_pagamento_dias": ("financeiro", lambda v: f"Prazo de pagamento de {v:.0f} dias"),
    "segmento": ("perfil", lambda v: f"Segmento {v}"),
    "cidade": ("entrega", lambda v: f"Cidade {v}"),
    "canal": ("relacionamento", lambda v: f"Atendido por {v}"),
    "vendedor": ("relacionamento", lambda v: f"Carteira do vendedor {v}"),
}
CAUSA_PARA_TEMA = {"servico": "entrega", "preco": "preco", "financeiro": "financeiro"}


def _agrupar_onehot(valores: np.ndarray, nomes: list[str]) -> pd.DataFrame:
    """Soma as colunas one-hot de volta no atributo original (ex.: cidade_Toledo -> cidade)."""
    df = pd.DataFrame(valores, columns=nomes)
    grupos = {n: next((c for c in CATEGORICAS if n.startswith(c + "_")), n) for n in nomes}
    return df.T.groupby(pd.Series(grupos)).sum().T[ATRIBUTOS]


def valores_shap(modelo: ModeloChurn, X: pd.DataFrame, fundo: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Contribuição de cada atributo (em log-odds) para o risco de cada cliente."""
    Xt = modelo.transformar(X)
    if isinstance(modelo.estimador, LGBMClassifier):
        sv = shap.TreeExplainer(modelo.estimador).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
    else:
        fundo_t = modelo.transformar(fundo.sample(min(200, len(fundo)), random_state=seed))

        def log_odds(a):
            p = np.clip(modelo.estimador.predict_proba(a)[:, 1], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))

        exp = shap.PermutationExplainer(log_odds, shap.maskers.Independent(fundo_t, max_samples=len(fundo_t)), seed=seed)
        sv = exp(Xt, max_evals=2 * Xt.shape[1] + 1, silent=True).values
    out = _agrupar_onehot(sv, modelo.nomes_transformados)
    out.index = X.index
    return out


def motivos(X: pd.DataFrame, shap_df: pd.DataFrame, n: int = 3, minimo: float = 0.15) -> pd.DataFrame:
    """Top-n motivos que AUMENTAM o risco (contribuição > `minimo` em log-odds), em texto, com o tema."""
    linhas = []
    for idx, contrib in shap_df.iterrows():
        top = contrib[contrib > minimo].sort_values(ascending=False).head(n)
        itens = [(DESCRICOES[a][1](X.at[idx, a]), DESCRICOES[a][0], float(v)) for a, v in top.items()]
        linha = {f"motivo_{i + 1}": t for i, (t, _, _) in enumerate(itens)}
        temas = list(dict.fromkeys(tema for _, tema, _ in itens))
        linha["temas"] = ";".join(temas)
        linha["tema_principal"] = next((t for t in temas if t not in ("engajamento", "perfil")),
                                       temas[0] if temas else "")
        linhas.append(linha)
    return pd.DataFrame(linhas, index=shap_df.index)


def avaliar_contra_gabarito(X: pd.DataFrame, mot: pd.DataFrame, verdade: pd.DataFrame) -> dict:
    """Só é possível porque os dados são simulados: o tema apontado bate com a causa real?"""
    df = X[["cliente_id"]].join(mot).merge(verdade, on="cliente_id")
    df = df[df["causa_churn"].isin(CAUSA_PARA_TEMA)]
    if df.empty:
        return {}
    acerto = df.apply(lambda r: CAUSA_PARA_TEMA[r.causa_churn] in r.temas.split(";"), axis=1)
    return {"clientes_avaliados": int(len(df)), "causa_real_entre_os_motivos": float(acerto.mean())}

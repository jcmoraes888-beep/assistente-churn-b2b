"""API REST do Assistente de Churn (FastAPI).

Rodar:  uvicorn churn_b2b.api:app --reload
Docs:   http://localhost:8000/docs
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from .agente.plano import MODELO_PADRAO, gerar_plano
from .rag.base import BaseConhecimento
from .rag.contexto import carregar_scores, contexto_cliente

app = FastAPI(title="Assistente de Churn B2B",
              description="Risco de churn, motivos (SHAP), políticas de retenção (RAG) e plano de ação (LLM). "
                          "Dados 100% fictícios.",
              version="0.3.0")


@lru_cache
def _scores():
    return carregar_scores().set_index("cliente_id")


@lru_cache
def _base():
    return BaseConhecimento()


def _limpo(d: dict) -> dict:
    """Converte NaN/numpy para JSON."""
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            v = v.item()
        out[k] = None if isinstance(v, float) and math.isnan(v) else v
    return out


def _cliente(cliente_id: str) -> dict:
    if cliente_id not in _scores().index:
        raise HTTPException(404, f"Cliente {cliente_id} não encontrado na carteira ativa.")
    return _limpo({"cliente_id": cliente_id, **_scores().loc[cliente_id].to_dict()})


@app.get("/saude")
def saude():
    return {"status": "ok", "clientes": int(len(_scores())), "embeddings": _base().emb.nome}


@app.get("/carteira")
def carteira(faixa: Literal["Alto", "Médio", "Baixo"] | None = None, tema: str | None = None,
             limite: int = Query(50, ge=1, le=2000)):
    df = _scores().reset_index()
    if faixa:
        df = df[df.faixa_risco == faixa]
    if tema:
        df = df[df.tema_principal == tema]
    df = df.sort_values("prob_churn", ascending=False).head(limite)
    return [_limpo(r) for r in df.to_dict("records")]


@app.get("/clientes/{cliente_id}")
def cliente(cliente_id: str):
    return _cliente(cliente_id)


@app.get("/clientes/{cliente_id}/politicas")
def politicas(cliente_id: str):
    _cliente(cliente_id)
    ctx = contexto_cliente(cliente_id, _base())
    return {"consulta": ctx["consulta"], "tema_principal": ctx["tema_principal"],
            "politicas": ctx["politicas"], "regras_gerais": ctx["regras_gerais"]}


@app.post("/clientes/{cliente_id}/plano")
def plano(cliente_id: str, modo: Literal["auto", "llm", "regras"] = "auto", modelo: str = MODELO_PADRAO):
    _cliente(cliente_id)
    try:
        r = gerar_plano(cliente_id, modo, _base(), modelo)
    except Exception as e:  # falha da API do LLM: devolve erro claro
        raise HTTPException(502, f"Falha ao gerar o plano com o LLM: {e}") from e
    return {"cliente_id": cliente_id, "modo": r["modo"], "modelo": r["modelo"],
            "plano": r["plano"].model_dump(), "validacao": r["validacao"],
            "politicas_usadas": [{"doc_id": t["doc_id"], "titulo": t["titulo"], "secao": t["secao"]}
                                 for t in r["contexto"]["politicas"]]}

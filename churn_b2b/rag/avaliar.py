"""Avalia o recuperador: para cada pergunta, a política certa aparece entre as 3 primeiras?

Métricas:
  acerto@1 / acerto@3  – % de perguntas com a política certa em 1º lugar / entre as 3 primeiras
  MRR                  – média de 1/posição da primeira política certa (1,0 = sempre em 1º)

Uso:  python -m churn_b2b.rag.avaliar                  # embeddings 'auto'
      python -m churn_b2b.rag.avaliar --comparar       # local x openai (se houver chave)
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from ..config import RAIZ, RELATORIOS
from .base import BaseConhecimento, indexar

ARQ = RAIZ / "politicas" / "perguntas_avaliacao.json"


def avaliar(provedor: str, usar_tema: bool, k: int = 3) -> tuple[dict, pd.DataFrame]:
    base = BaseConhecimento(provedor)
    linhas = []
    for q in json.loads(ARQ.read_text("utf-8")):
        tema = q["tema"] if usar_tema and q["tema"] not in ("geral", "perfil") else None
        docs = []
        for r in base.buscar(q["pergunta"], k=10, tema=tema):   # busca mais e deduplica por política
            if r["doc_id"] not in docs:
                docs.append(r["doc_id"])
        pos = next((i + 1 for i, d in enumerate(docs) if d in q["esperado"]), None)
        linhas.append({"pergunta": q["pergunta"], "esperado": "/".join(q["esperado"]),
                       "top3": ", ".join(docs[:k]), "posicao": pos})
    df = pd.DataFrame(linhas)
    met = {"acerto@1": (df.posicao == 1).mean(), "acerto@3": (df.posicao <= k).mean(),
           "MRR": df.posicao.apply(lambda p: 1 / p if p else 0).mean()}
    return met, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="auto", choices=["auto", "openai", "local"])
    ap.add_argument("--comparar", action="store_true", help="avalia local e openai lado a lado")
    a = ap.parse_args()

    provedores = ["local", "openai"] if a.comparar else [a.embeddings]
    if a.comparar and not os.environ.get("OPENAI_API_KEY"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        if not os.environ.get("OPENAI_API_KEY"):
            provedores = ["local"]
            print("[aviso] sem OPENAI_API_KEY: avaliando só o embedding local.")

    tabela, erros = [], None
    for p in provedores:
        indexar(p)
        for usar_tema in (False, True):
            met, df = avaliar(p, usar_tema)
            tabela.append({"embeddings": p, "filtro_por_tema": "sim" if usar_tema else "não",
                           **{k: round(v, 3) for k, v in met.items()}})
            if usar_tema:
                erros = df[(df.posicao.isna()) | (df.posicao > 1)]
    res = pd.DataFrame(tabela)
    print("\n" + res.to_string(index=False))
    if erros is not None and len(erros):
        print("\nPerguntas em que a política certa não ficou em 1º lugar (último cenário):")
        print(erros[["pergunta", "esperado", "top3"]].to_string(index=False))
    RELATORIOS.mkdir(exist_ok=True)
    (RELATORIOS / "avaliacao_rag.md").write_text(
        "# Avaliação do RAG\n\n" + f"{len(json.loads(ARQ.read_text('utf-8')))} perguntas com a política "
        "esperada definida à mão (`politicas/perguntas_avaliacao.json`).\n\n" + res.to_markdown(index=False),
        "utf-8")


if __name__ == "__main__":
    main()

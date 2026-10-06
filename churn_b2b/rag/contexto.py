"""Junta tudo para UM cliente: risco do modelo + motivos SHAP + políticas recuperadas.

É exatamente o pacote que a Etapa 3 vai entregar ao LLM.

Uso:  python -m churn_b2b.rag.contexto C00700
      python -m churn_b2b.rag.contexto            # mostra o cliente de maior risco
"""
from __future__ import annotations

import argparse

import pandas as pd

from ..config import DADOS
from .base import BaseConhecimento


def montar_consulta(c: pd.Series) -> str:
    """Pergunta em linguagem natural, no vocabulário das políticas."""
    motivos = "; ".join(m for m in [c.get("motivo_1"), c.get("motivo_2"), c.get("motivo_3")]
                        if isinstance(m, str) and m)
    return (f"Cliente do segmento {c['segmento']} em {c['cidade']}, atendido por {c['canal']}, "
            f"com risco {str(c['faixa_risco']).lower()} de parar de comprar. Sinais: {motivos}. "
            f"Que ações de retenção a política permite?")


def contexto_cliente(cliente_id: str | None = None, base: BaseConhecimento | None = None, k: int = 3) -> dict:
    scores = pd.read_csv(DADOS / "scores_atuais.csv")
    c = scores.iloc[0] if cliente_id is None else scores.set_index("cliente_id").loc[cliente_id].copy()
    if cliente_id:
        c["cliente_id"] = cliente_id
    base = base or BaseConhecimento()
    consulta = montar_consulta(c)
    tema = c["tema_principal"] if isinstance(c.get("tema_principal"), str) and c["tema_principal"] else None

    trechos = base.buscar(consulta, k=k, tema=tema, segmento=c["segmento"])
    # segundo tema também entra (ex.: financeiro + engajamento)
    temas = [t for t in str(c.get("temas", "")).split(";") if t and t != tema and t != "perfil"]
    if temas:
        for t in base.buscar(consulta, k=1, tema=temas[0], segmento=c["segmento"]):
            if t["id"] not in {x["id"] for x in trechos}:
                trechos.append(t)
    return {"cliente": c.to_dict(), "consulta": consulta, "tema_principal": tema,
            "politicas": trechos, "regras_gerais": base.regras_gerais()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cliente_id", nargs="?")
    ap.add_argument("--embeddings", default="auto", choices=["auto", "openai", "local"])
    a = ap.parse_args()
    ctx = contexto_cliente(a.cliente_id, BaseConhecimento(a.embeddings))
    c = ctx["cliente"]
    print(f"\n{c['cliente_id']} · {c['segmento']} · {c['cidade']} · risco {c['faixa_risco']} "
          f"({c['prob_churn']:.0%})")
    for i in (1, 2, 3):
        if isinstance(c.get(f"motivo_{i}"), str):
            print(f"  • {c[f'motivo_{i}']}")
    print(f"\nTema principal: {ctx['tema_principal']}\nConsulta: {ctx['consulta']}\n")
    for t in ctx["politicas"]:
        print(f"[{t['doc_id']}] {t['titulo']} › {t['secao']}  (similaridade {t['similaridade']})")
    print(f"\n+ {len(ctx['regras_gerais'])} trechos das regras gerais (POL-10) sempre incluídos.")


if __name__ == "__main__":
    main()

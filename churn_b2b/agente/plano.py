"""Etapa 3 — Agente de retenção: transforma (risco + motivos + políticas) em um plano de ação.

Dois modos:
  llm     LangChain + OpenAI com saída estruturada (Pydantic). Usa OPENAI_API_KEY do .env.
  regras  Plano montado por regras a partir das mesmas políticas. Sem API e sem custo:
          serve para demonstração, testes automáticos e como plano B se a API falhar.

Uso:  python -m churn_b2b.agente.plano C00700            # LLM (se houver chave)
      python -m churn_b2b.agente.plano C00700 --modo regras
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from ..rag.base import BaseConhecimento
from ..rag.contexto import contexto_cliente
from .validacao import validar_plano

MODELO_PADRAO = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
Responsavel = Literal["vendedor", "supervisor comercial", "gerente comercial", "financeiro", "logística"]


# ----------------------------------------------------------------------------- formato da resposta
class Acao(BaseModel):
    descricao: str = Field(description="O que fazer, de forma concreta e verificável")
    responsavel: Responsavel
    prazo: str = Field(description="Ex.: 'em até 48 horas', 'até o dia 7'")
    politica: str = Field(description="Id da política que autoriza a ação, ex.: POL-02")
    condicao_comercial: str | None = Field(
        default=None, description="Desconto, prazo ou bonificação oferecida, com o número exato; null se não houver")


class PlanoRetencao(BaseModel):
    diagnostico: str = Field(description="2 a 3 frases explicando, em linguagem de negócio, por que o cliente está em risco")
    prioridade: Literal["Alta", "Média", "Baixa"]
    prazo_primeiro_contato: str
    acoes: list[Acao] = Field(description="De 2 a 4 ações, na ordem em que devem acontecer")
    roteiro_ligacao: list[str] = Field(description="3 a 5 perguntas abertas para o primeiro contato")
    mensagem_cliente: str = Field(
        description="Mensagem curta de WhatsApp do vendedor para o cliente. Cordial, sem citar risco, churn ou modelo")
    o_que_evitar: list[str] = Field(description="1 a 3 cuidados com base nas políticas")
    politicas_citadas: list[str]


# ------------------------------------------------------------------------------------- prompt
SISTEMA = """Você é analista de retenção de clientes da Distribuidora Modelo Oeste (empresa fictícia de \
alimentos no oeste do Paraná). Recebe um cliente com risco de parar de comprar, os motivos apontados pelo \
modelo de machine learning e os trechos das políticas internas da empresa.

Regras obrigatórias:
1. Use SOMENTE ações permitidas pelas políticas fornecidas. Cada ação cita o id da política (ex.: POL-02).
2. Respeite as alçadas e os números das políticas (percentuais, prazos, valores). Nunca invente condições. \
Alçada de desconto (POL-01): vendedor até 3%, supervisor comercial até 6%, gerente comercial até 10%. \
O responsável pela ação tem que ter alçada para o percentual oferecido.
3. Primeiro resolva a causa (entrega, financeiro, produto, relacionamento). Condição comercial só depois e \
dentro da alçada.
4. A mensagem ao cliente NUNCA menciona risco, churn, modelo, classificação ou "sistema". Tom de parceria, \
português do Brasil, no máximo 5 linhas, assinada exatamente como [Nome do vendedor]. Nunca use o código \
interno do vendedor (ex.: V03) nem o código do cliente na mensagem.
5. Se um motivo não tiver política correspondente nos trechos, diga isso no diagnóstico em vez de inventar.
6. A prioridade segue a faixa de risco (Alto → Alta, Médio → Média, Baixo → Baixa).
7. Sempre que a ação envolver desconto, prazo, parcelamento, bonificação ou frete, escreva a condição exata (com número) no campo condicao_comercial, e não só na descrição."""

HUMANO = """## Cliente
{cliente}

## Motivos apontados pelo modelo (SHAP)
{motivos}

## Políticas recuperadas
{politicas}

## Regras gerais (valem sempre)
{regras}

Monte o plano de retenção."""


def formatar_risco(p: float) -> str:
    """Probabilidades extremas viram faixa ("mais de 95%"): evita a falsa precisão de "100%"."""
    p = float(p)
    return "mais de 95%" if p > 0.95 else ("menos de 5%" if p < 0.05 else f"{p:.0%}")


def _fmt_trechos(trechos: list[dict]) -> str:
    return "\n\n".join(f"[{t['doc_id']}] {t['texto']}" for t in trechos)


def montar_entrada(ctx: dict) -> dict:
    c = ctx["cliente"]
    cliente = (f"{c['cliente_id']} · {c['razao_social']} · segmento {c['segmento']} · {c['cidade']} · "
               f"canal {c['canal']} · vendedor {c['vendedor']}\n"
               f"Faturamento últimos 90 dias: R$ {float(c['receita_90d']):,.2f} · "
               f"Última compra há {int(c['recencia_dias'])} dias\n"
               f"Risco de parar de comprar nos próximos 90 dias: {formatar_risco(c['prob_churn'])} "
               f"(faixa {c['faixa_risco']})").replace(",", "X").replace(".", ",").replace("X", ".")
    motivos = "\n".join(f"- {c[f'motivo_{i}']}" for i in (1, 2, 3) if isinstance(c.get(f"motivo_{i}"), str))
    return {"cliente": cliente, "motivos": motivos or "- (sem motivos destacados)",
            "politicas": _fmt_trechos(ctx["politicas"]), "regras": _fmt_trechos(ctx["regras_gerais"])}


def _cadeia(modelo: str):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    prompt = ChatPromptTemplate.from_messages([("system", SISTEMA), ("human", HUMANO)])
    llm = ChatOpenAI(model=modelo, temperature=0.2, timeout=90, max_retries=2)
    return prompt | llm.with_structured_output(PlanoRetencao)


# --------------------------------------------------------------------- plano por regras (sem API)
_ACOES_POR_TEMA = {
    "financeiro": [("Contato do financeiro junto com o vendedor para entender o momento do cliente e "
                    "propor reprogramação do vencimento", "financeiro", "em até 2 dias úteis", "POL-02",
                    "Reprogramação do vencimento em até 15 dias, sem juros")],
    "entrega": [("Ligação do supervisor de logística explicando a causa dos atrasos e o plano de correção",
                 "logística", "em até 2 dias úteis", "POL-03", None),
                ("Incluir o cliente na rota prioritária por 60 dias e oferecer janela de entrega agendada",
                 "logística", "a partir da próxima entrega", "POL-03", "Bonificação de 1% por entrega atrasada (máx. 4%)")],
    "preco": [("Pedir a cotação do concorrente por escrito e comparar nos 20 itens de maior giro do cliente",
               "vendedor", "na primeira visita", "POL-01", None),
              ("Oferecer desconto adicional dentro da alçada do vendedor, com contrapartida registrada no CRM",
               "vendedor", "em até 7 dias", "POL-01", "Até 3% por 60 dias, com contrapartida de volume ou mix")],
    "produto": [("Revisar com o cliente os itens que ele deixou de comprar e perguntar onde está comprando",
                 "vendedor", "na primeira visita", "POL-05", None),
                ("Reservar estoque dos 10 itens de maior giro do cliente", "vendedor", "imediato", "POL-05", None)],
    "relacionamento": [("Visita do supervisor junto com o vendedor", "supervisor comercial", "em até 7 dias",
                        "POL-06", None)],
    "canal": [("Designar vendedor externo para visita, mantendo o canal atual para pedidos", "supervisor comercial",
               "em até 7 dias", "POL-09", None)],
    "engajamento": [("Ligação do vendedor perguntando abertamente se houve algum problema", "vendedor",
                     "em até 48 horas", "POL-04", None),
                    ("Se não houver pedido em 30 dias, oferecer a condição de reativação", "vendedor", "dia 30",
                     "POL-04", "Frete grátis nos 2 próximos pedidos")],
}
_MENSAGEM = {
    "financeiro": "Vi que os últimos boletos ficaram um pouco apertados e quero entender como podemos ajudar. "
                  "Temos algumas opções de prazo que podem facilitar.",
    "entrega": "Sei que algumas entregas recentes atrasaram e peço desculpas por isso. Já estamos corrigindo "
               "a rota e queria combinar o melhor horário de entrega para vocês.",
    "preco": "Queria marcar uma conversa rápida para revisar os preços dos itens que vocês mais compram.",
    "produto": "Queria revisar com você o mix dos últimos pedidos e garantir que não falte nada do que vocês mais usam.",
}


def plano_por_regras(ctx: dict) -> PlanoRetencao:
    c = ctx["cliente"]
    recuperadas = {t["doc_id"] for t in ctx["politicas"]}
    temas = [ctx.get("tema_principal")] + [t for t in str(c.get("temas", "")).split(";")]
    acoes = []
    for tema in dict.fromkeys(t for t in temas if t):
        for desc, resp, prazo, pol, cond in _ACOES_POR_TEMA.get(tema, []):
            if pol in recuperadas and len(acoes) < 4:
                acoes.append(Acao(descricao=desc, responsavel=resp, prazo=prazo, politica=pol, condicao_comercial=cond))
    if not acoes:
        acoes.append(Acao(descricao="Contato do vendedor para ouvir o cliente e confirmar o motivo da queda",
                          responsavel="vendedor", prazo="em até 48 horas", politica="POL-10"))
    faixa = c["faixa_risco"]
    motivos = [c[f"motivo_{i}"] for i in (1, 2, 3) if isinstance(c.get(f"motivo_{i}"), str)]
    return PlanoRetencao(
        diagnostico=f"{c['segmento']} de {c['cidade']} com sinais de afastamento: "
                    f"{'; '.join(motivos) or 'queda de compras'}.",
        prioridade={"Alto": "Alta", "Médio": "Média"}.get(faixa, "Baixa"),
        prazo_primeiro_contato={"Alto": "em até 48 horas", "Médio": "em até 7 dias"}.get(faixa, "acompanhamento normal"),
        acoes=acoes,
        roteiro_ligacao=["Como está o movimento por aí neste mês?",
                         "Teve algum problema com pedidos, entregas ou boletos recentemente?",
                         "Tem algum produto que vocês estão comprando de outro fornecedor? Por quê?",
                         "O que faria vocês voltarem a concentrar as compras com a gente?"],
        mensagem_cliente=f"Olá! Tudo bem? Aqui é [Nome do vendedor], da Distribuidora Modelo Oeste. "
                         f"{_MENSAGEM.get(ctx.get('tema_principal') or '', 'Senti falta dos seus pedidos e queria saber como estão as coisas por aí.')} "
                         f"Posso te ligar hoje ou amanhã?",
        o_que_evitar=["Não mencionar ao cliente qualquer classificação de risco (POL-10)",
                      "Não oferecer condição comercial fora das alçadas (POL-01)"],
        politicas_citadas=sorted({a.politica for a in acoes} | {"POL-10"}),
    )


# --------------------------------------------------------------------------------- principal
def gerar_plano(cliente_id: str, modo: str = "auto", base: BaseConhecimento | None = None,
                modelo: str = MODELO_PADRAO) -> dict:
    """Retorna contexto + plano + validação. modo: auto | llm | regras."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ctx = contexto_cliente(cliente_id, base)
    if modo == "auto":
        modo = "llm" if os.environ.get("OPENAI_API_KEY") else "regras"
    if modo == "llm":
        plano = _cadeia(modelo).invoke(montar_entrada(ctx))
    else:
        plano = plano_por_regras(ctx)
    return {"modo": modo, "modelo": modelo if modo == "llm" else None, "contexto": ctx,
            "plano": plano, "validacao": validar_plano(plano, ctx)}


def imprimir(r: dict) -> None:
    c, p = r["contexto"]["cliente"], r["plano"]
    print(f"\n{c['cliente_id']} · {c['segmento']} · {c['cidade']} · risco {c['faixa_risco']} "
          f"· plano gerado por: {r['modo']}{' (' + r['modelo'] + ')' if r['modelo'] else ''}\n")
    print(f"Diagnóstico: {p.diagnostico}\nPrioridade: {p.prioridade} · primeiro contato {p.prazo_primeiro_contato}\n")
    for i, a in enumerate(p.acoes, 1):
        print(f"{i}. [{a.politica}] {a.descricao} — {a.responsavel}, {a.prazo}"
              + (f"\n   Condição: {a.condicao_comercial}" if a.condicao_comercial else ""))
    print(f"\nMensagem para o cliente:\n{p.mensagem_cliente}\n")
    if r["validacao"]:
        print("Validação automática:")
        for v in r["validacao"]:
            print(f"  {v['nivel']} {v['mensagem']}")
    else:
        print("Validação automática: ✅ nenhum problema encontrado")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cliente_id")
    ap.add_argument("--modo", default="auto", choices=["auto", "llm", "regras"])
    ap.add_argument("--modelo", default=MODELO_PADRAO)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = gerar_plano(a.cliente_id, a.modo, modelo=a.modelo)
    if a.json:
        print(json.dumps(r["plano"].model_dump(), ensure_ascii=False, indent=2))
    else:
        imprimir(r)


if __name__ == "__main__":
    main()

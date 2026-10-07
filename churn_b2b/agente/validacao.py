"""Validação automática do plano gerado (guardrails).

O LLM escreve bem, mas pode inventar. Estas regras conferem o plano contra as políticas
antes de ele chegar ao vendedor.
"""
from __future__ import annotations

import re
import unicodedata

ERRO, ALERTA = "🔴", "🟠"
PROIBIDAS_NA_MENSAGEM = ["churn", "risco", "classificad", "modelo", "algoritmo", "inteligencia artificial",
                         "sistema apontou", "previsao"]
ALCADA = {"vendedor": 3.0, "supervisor comercial": 6.0, "gerente comercial": 10.0}


def _sem_acento(t: str) -> str:
    return unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()


def _percentuais_desconto(texto: str) -> list[float]:
    """Percentuais que aparecem perto da palavra desconto."""
    t = _sem_acento(texto or "")
    if "desconto" not in t:
        return []
    # set(): o mesmo 7% citado na descrição e na condição comercial conta uma vez só
    return sorted({float(x.replace(",", ".")) for x in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", t)})


def validar_plano(plano, ctx: dict) -> list[dict]:
    achados = []

    def add(nivel, msg):
        achados.append({"nivel": nivel, "mensagem": msg})

    permitidas = {t["doc_id"] for t in ctx["politicas"]} | {t["doc_id"] for t in ctx["regras_gerais"]}

    # 1) toda ação precisa citar uma política que foi de fato recuperada
    for a in plano.acoes:
        if a.politica not in permitidas:
            add(ERRO, f"A ação \"{a.descricao[:60]}…\" cita {a.politica}, que não estava nos trechos "
                      "recuperados (possível invenção).")

    # 2) alçadas de desconto (POL-01)
    for a in plano.acoes:
        for pct in _percentuais_desconto(f"{a.descricao} {a.condicao_comercial or ''}"):
            if pct > 10:
                add(ERRO, f"Desconto de {pct:g}% acima do teto de 10% da POL-01.")
            elif pct > ALCADA.get(a.responsavel, 0):
                add(ALERTA, f"Desconto de {pct:g}% está acima da alçada de '{a.responsavel}' "
                            f"(até {ALCADA.get(a.responsavel, 0):g}%). Precisa de aprovação superior.")

    # 3) inadimplente não recebe desconto antes de regularizar (POL-02)
    if ctx.get("tema_principal") == "financeiro":
        for a in plano.acoes:
            if _percentuais_desconto(f"{a.descricao} {a.condicao_comercial or ''}"):
                add(ALERTA, "Cliente com tema financeiro recebeu oferta de desconto: a POL-02 manda regularizar "
                            "a situação financeira antes.")

    # 4) a mensagem não pode expor a classificação de risco (POL-10)
    msg = _sem_acento(plano.mensagem_cliente).replace("distribuidora modelo oeste", "")
    for termo in PROIBIDAS_NA_MENSAGEM:
        if termo in msg:
            add(ERRO, f"A mensagem ao cliente menciona '{termo}'. A POL-10 proíbe expor a classificação.")

    # 4b) códigos internos não podem aparecer para o cliente
    if re.search(r"\b(V\d{2}|C\d{5})\b", plano.mensagem_cliente):
        add(ALERTA, "A mensagem ao cliente traz um código interno (vendedor ou cliente). "
                    "Troque pelo nome do vendedor antes de enviar.")

    # 5) prioridade coerente com a faixa de risco
    esperado = {"Alto": "Alta", "Médio": "Média", "Baixo": "Baixa"}.get(ctx["cliente"]["faixa_risco"])
    if esperado and plano.prioridade != esperado:
        add(ALERTA, f"Prioridade '{plano.prioridade}' diferente da faixa de risco ({ctx['cliente']['faixa_risco']}).")

    # 6) mensagem curta
    if len(plano.mensagem_cliente) > 700:
        add(ALERTA, "Mensagem ao cliente longa demais para WhatsApp.")

    # sem repetição: duas ações com o mesmo problema geram um aviso só
    vistos, unicos = set(), []
    for a in achados:
        if a["mensagem"] not in vistos:
            vistos.add(a["mensagem"])
            unicos.append(a)
    return unicos

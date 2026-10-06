"""Carrega as políticas (Markdown com cabeçalho) e fatia por seção (## ...)."""
from __future__ import annotations

import re
from pathlib import Path

from ..config import RAIZ

PASTA_POLITICAS = RAIZ / "politicas"
TEMAS = ["preco", "financeiro", "entrega", "engajamento", "produto", "relacionamento", "canal", "perfil", "geral"]
SEGMENTOS = ["Supermercado", "Mercearia", "Padaria", "Restaurante", "Lanchonete", "Hotel"]


def _cabecalho(texto: str) -> tuple[dict, str]:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", texto, flags=re.S)
    if not m:
        raise ValueError("Política sem cabeçalho (--- id/titulo/temas/segmentos ---)")
    meta = dict(linha.split(":", 1) for linha in m.group(1).splitlines() if ":" in linha)
    return {k.strip(): v.strip() for k, v in meta.items()}, m.group(2)


def carregar_trechos(pasta: Path = PASTA_POLITICAS) -> list[dict]:
    """Um trecho por seção. Cada trecho leva o título da política para não perder o contexto."""
    trechos = []
    for arq in sorted(pasta.glob("*.md")):
        meta, corpo = _cabecalho(arq.read_text(encoding="utf-8").replace("\r\n", "\n"))
        temas = [t.strip() for t in meta["temas"].split(";")]
        segs = [s.strip() for s in meta["segmentos"].split(";")]
        partes = re.split(r"^## ", corpo, flags=re.M)[1:]   # descarta o que vem antes da 1ª seção
        for i, parte in enumerate(partes, 1):
            secao, _, texto = parte.partition("\n")
            trechos.append({
                "id": f"{meta['id']}-{i:02d}",
                "doc_id": meta["id"],
                "titulo": meta["titulo"],
                "secao": secao.strip(),
                "texto": f"{meta['titulo']} — {secao.strip()}\n{texto.strip()}",
                "temas": temas,
                "segmentos": segs,
                "arquivo": arq.name,
            })
    return trechos


def metadados(t: dict) -> dict:
    """Metadados planos para o ChromaDB (aceita só str/int/float/bool)."""
    md = {"doc_id": t["doc_id"], "titulo": t["titulo"], "secao": t["secao"], "arquivo": t["arquivo"],
          "temas": ";".join(t["temas"]), "segmentos": ";".join(t["segmentos"])}
    md.update({f"tema_{x}": x in t["temas"] for x in TEMAS})
    todos = "todos" in t["segmentos"]
    md["seg_todos"] = todos
    md.update({f"seg_{s}": todos or s in t["segmentos"] for s in SEGMENTOS})
    return md

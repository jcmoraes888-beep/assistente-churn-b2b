"""Indexação no ChromaDB e busca das políticas.

Uso:
  python -m churn_b2b.rag.base                      # (re)indexa as políticas
  python -m churn_b2b.rag.base --embeddings local   # sem API
"""
from __future__ import annotations

import argparse

import chromadb

from ..config import DADOS
from . import embeddings as E
from .documentos import carregar_trechos, metadados

PASTA_CHROMA = DADOS / "chroma"


def _colecao(emb, criar: bool = False):
    cliente = chromadb.PersistentClient(path=str(PASTA_CHROMA))
    nome = f"politicas_{emb.nome}"
    if criar:
        try:
            cliente.delete_collection(nome)
        except Exception:
            pass
        return cliente.create_collection(nome, metadata={"hnsw:space": "cosine"})
    return cliente.get_collection(nome)


def indexar(provedor: str = "auto") -> int:
    emb = E.obter(provedor)
    trechos = carregar_trechos()
    col = _colecao(emb, criar=True)
    col.add(ids=[t["id"] for t in trechos], documents=[t["texto"] for t in trechos],
            metadatas=[metadados(t) for t in trechos], embeddings=emb([t["texto"] for t in trechos]))
    return len(trechos)


class BaseConhecimento:
    """Busca semântica nas políticas, com filtros opcionais por tema e segmento."""

    def __init__(self, provedor: str = "auto"):
        self.emb = E.obter(provedor)
        try:
            self.col = _colecao(self.emb)
        except Exception:
            indexar(self.emb.nome)
            self.col = _colecao(self.emb)

    def buscar(self, consulta: str, k: int = 3, tema: str | None = None,
               segmento: str | None = None) -> list[dict]:
        filtros = []
        if tema:
            filtros.append({f"tema_{tema}": True})
        if segmento:
            filtros.append({f"seg_{segmento}": True})
        where = None if not filtros else (filtros[0] if len(filtros) == 1 else {"$and": filtros})

        r = self.col.query(query_embeddings=self.emb([consulta]), n_results=k, where=where)
        if where and len(r["ids"][0]) < k:   # filtro restritivo demais: completa sem filtro
            extra = self.col.query(query_embeddings=self.emb([consulta]), n_results=k)
            for chave in ("ids", "documents", "metadatas", "distances"):
                r[chave][0] += [x for i, x in enumerate(extra[chave][0]) if extra["ids"][0][i] not in r["ids"][0]]
        return [{"id": i, "doc_id": m["doc_id"], "titulo": m["titulo"], "secao": m["secao"],
                 "texto": d, "similaridade": round(1 - dist, 3)}
                for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])][:k]

    def regras_gerais(self) -> list[dict]:
        """A POL-10 vale para todos os casos e vai sempre para o LLM."""
        r = self.col.get(where={"doc_id": "POL-10"})
        return [{"id": i, "doc_id": m["doc_id"], "titulo": m["titulo"], "secao": m["secao"], "texto": d}
                for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="auto", choices=["auto", "openai", "local"])
    a = ap.parse_args()
    n = indexar(a.embeddings)
    print(f"{n} trechos de {len(set(t['doc_id'] for t in carregar_trechos()))} políticas indexados "
          f"em {PASTA_CHROMA} (embeddings: {E.obter(a.embeddings).nome})")

"""Dois provedores de embeddings:

- openai: text-embedding-3-small (melhor qualidade; precisa de OPENAI_API_KEY no .env).
- local:  vetor de n-gramas de caracteres (sem internet e sem custo). Serve para testes, CI e
          para comparar com o OpenAI na avaliação. Não entende sinônimos, só palavras parecidas.
"""
from __future__ import annotations

import os
import unicodedata

import numpy as np

MODELO_OPENAI = "text-embedding-3-small"


def _normalizar(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()
    return t


class EmbeddingLocal:
    nome = "local"

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 12,
                                     alternate_sign=False, norm="l2")

    def __call__(self, textos: list[str]) -> list[list[float]]:
        return self.vec.transform([_normalizar(t) for t in textos]).toarray().astype(np.float32).tolist()


class EmbeddingOpenAI:
    nome = "openai"

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI
        chave = api_key or os.environ.get("OPENAI_API_KEY")
        if not chave:
            raise RuntimeError("OPENAI_API_KEY não encontrada. Crie um arquivo .env com OPENAI_API_KEY=sk-... "
                               "ou use --embeddings local.")
        self.cliente = OpenAI(api_key=chave)

    def __call__(self, textos: list[str]) -> list[list[float]]:
        resp = self.cliente.embeddings.create(model=MODELO_OPENAI, input=textos)
        return [d.embedding for d in resp.data]


def obter(provedor: str = "auto"):
    """'auto' usa OpenAI se houver chave; senão, o local."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if provedor == "auto":
        provedor = os.environ.get("CHURN_EMBEDDINGS") or ("openai" if os.environ.get("OPENAI_API_KEY") else "local")
    return EmbeddingOpenAI() if provedor == "openai" else EmbeddingLocal()

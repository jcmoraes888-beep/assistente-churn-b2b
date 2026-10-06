"""Testes do RAG com embeddings locais (sem internet e sem custo)."""
import pytest

from churn_b2b.rag import base as B
from churn_b2b.rag.avaliar import avaliar
from churn_b2b.rag.documentos import carregar_trechos, metadados


@pytest.fixture(autouse=True)
def chroma_temporario(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "PASTA_CHROMA", tmp_path / "chroma")


def test_politicas_carregadas():
    trechos = carregar_trechos()
    assert len({t["doc_id"] for t in trechos}) == 10
    assert all(t["texto"].startswith(t["titulo"]) for t in trechos)
    md = metadados(next(t for t in trechos if t["doc_id"] == "POL-07"))
    assert md["seg_Restaurante"] and not md["seg_Supermercado"]


def test_filtro_por_tema_e_segmento():
    B.indexar("local")
    kb = B.BaseConhecimento("local")
    r = kb.buscar("cliente pediu mais desconto", k=3, tema="preco", segmento="Supermercado")
    assert r and all(x["doc_id"] in {"POL-01", "POL-08"} for x in r)
    assert {x["doc_id"] for x in kb.regras_gerais()} == {"POL-10"}


def test_qualidade_minima_da_busca():
    B.indexar("local")
    met, _ = avaliar("local", usar_tema=True)
    assert met["acerto@3"] >= 0.9

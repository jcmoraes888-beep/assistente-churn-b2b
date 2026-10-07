"""Testes do agente e da API sem chamar o LLM (modo regras e LLM simulado)."""
import pytest
from fastapi.testclient import TestClient

from churn_b2b.agente import plano as P
from churn_b2b.agente.validacao import validar_plano
from churn_b2b.rag import base as B
from churn_b2b.rag.contexto import contexto_cliente


@pytest.fixture(autouse=True)
def ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "PASTA_CHROMA", tmp_path / "chroma")
    monkeypatch.setenv("CHURN_EMBEDDINGS", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def ctx():
    from churn_b2b.rag.contexto import carregar_scores
    cid = carregar_scores().query("faixa_risco == 'Alto' and tema_principal == 'financeiro'").cliente_id.iloc[0]
    return contexto_cliente(cid, B.BaseConhecimento("local"))


def test_plano_por_regras_passa_na_validacao(ctx):
    plano = P.plano_por_regras(ctx)
    assert plano.prioridade == "Alta"
    assert any(a.politica == "POL-02" for a in plano.acoes)
    assert validar_plano(plano, ctx) == []


def test_validacao_pega_plano_ruim(ctx):
    ruim = P.PlanoRetencao(
        diagnostico="x", prioridade="Baixa", prazo_primeiro_contato="quando der",
        acoes=[P.Acao(descricao="Dar desconto de 12% no portfólio todo", responsavel="vendedor",
                      prazo="hoje", politica="POL-99", condicao_comercial="12% de desconto")],
        roteiro_ligacao=["?"], mensagem_cliente="Nosso sistema apontou que você está em risco de churn.",
        o_que_evitar=[], politicas_citadas=["POL-99"])
    textos = " ".join(v["mensagem"] for v in validar_plano(ruim, ctx))
    for esperado in ["POL-99", "teto de 10%", "POL-02", "churn", "Prioridade"]:
        assert esperado in textos


def test_llm_simulado_e_entrada_do_prompt(ctx, monkeypatch):
    from langchain_core.runnables import RunnableLambda
    recebido = {}

    def falso_llm(entrada):
        recebido.update(entrada)
        return P.plano_por_regras(ctx)

    monkeypatch.setattr(P, "_cadeia", lambda modelo: RunnableLambda(falso_llm))
    r = P.gerar_plano(ctx["cliente"]["cliente_id"], "llm", B.BaseConhecimento("local"))
    assert r["modo"] == "llm"
    assert "[POL-02]" in recebido["politicas"] and "[POL-10]" in recebido["regras"]


def test_api():
    from churn_b2b import api
    api._scores.cache_clear(); api._base.cache_clear()
    c = TestClient(api.app)
    assert c.get("/saude").json()["status"] == "ok"
    alto = c.get("/carteira", params={"faixa": "Alto", "limite": 5}).json()
    assert len(alto) == 5 and all(x["faixa_risco"] == "Alto" for x in alto)
    cid = alto[0]["cliente_id"]
    r = c.post(f"/clientes/{cid}/plano", params={"modo": "regras"}).json()
    assert r["modo"] == "regras" and r["plano"]["acoes"]
    assert c.get("/clientes/XXXXX").status_code == 404


def test_validacao_pega_codigo_interno_na_mensagem(ctx):
    plano = P.plano_por_regras(ctx).model_copy(update={"mensagem_cliente": "Olá! Posso ajudar? Abraço, V03"})
    assert any("código interno" in v["mensagem"] for v in validar_plano(plano, ctx))

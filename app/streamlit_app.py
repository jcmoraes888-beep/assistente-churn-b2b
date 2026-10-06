"""Tela do Assistente de Churn B2B.

Rodar:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # permite importar churn_b2b

st.set_page_config(page_title="Assistente de Churn B2B", page_icon="📉", layout="wide")

# chave da OpenAI: secrets do Streamlit Cloud, .env ou variável de ambiente
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from churn_b2b.agente.plano import MODELO_PADRAO, formatar_risco, gerar_plano  # noqa: E402
from churn_b2b.rag.base import BaseConhecimento  # noqa: E402
from churn_b2b.rag.contexto import carregar_scores  # noqa: E402

TEMAS = {"financeiro": "💰 Financeiro", "entrega": "🚚 Entrega", "preco": "🏷️ Preço", "produto": "📦 Produto",
         "engajamento": "📉 Engajamento", "relacionamento": "🤝 Relacionamento", "canal": "🛒 Canal",
         "perfil": "🏪 Perfil"}
TEM_CHAVE = bool(os.environ.get("OPENAI_API_KEY"))


def brl(v: float) -> str:
    return "R$ " + f"{v:,.0f}".replace(",", ".")


@st.cache_data
def scores() -> pd.DataFrame:
    return carregar_scores()


@st.cache_resource(show_spinner="Preparando a base de políticas…")
def base() -> BaseConhecimento:
    return BaseConhecimento()


# ------------------------------------------------------------------------------------ barra lateral
df = scores()
with st.sidebar:
    st.header("📉 Assistente de Churn")
    st.caption("Distribuidora Modelo Oeste · dados 100% fictícios")
    faixas = st.multiselect("Faixa de risco", ["Alto", "Médio", "Baixo"], default=["Alto"])
    temas = st.multiselect("Tema principal", sorted(df.tema_principal.dropna().unique()),
                           format_func=lambda t: TEMAS.get(t, t))
    segmentos = st.multiselect("Segmento", sorted(df.segmento.unique()))
    st.divider()
    opcoes = (["IA (OpenAI)"] if TEM_CHAVE else []) + ["Regras (sem custo)"]
    modo_nome = st.radio("Plano gerado por", opcoes,
                         help="IA: LangChain + OpenAI com saída estruturada. Regras: mesmo pipeline, sem API.")
    modo = "llm" if modo_nome.startswith("IA") else "regras"
    modelo = st.text_input("Modelo", MODELO_PADRAO) if modo == "llm" else MODELO_PADRAO
    if not TEM_CHAVE:
        st.info("Sem chave da OpenAI: o plano é montado por regras a partir das mesmas políticas.")

# --------------------------------------------------------------------------------------- topo
st.title("Assistente de Churn B2B")
st.caption("Quem vai parar de comprar · por quê (SHAP) · o que a política permite (RAG) · plano de ação (LLM)")

alto = df[df.faixa_risco == "Alto"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Clientes ativos", f"{len(df):,}".replace(",", "."))
k2.metric("Em risco alto", len(alto), f"{len(alto) / len(df):.0%} da carteira", delta_color="off")
k3.metric("Faturamento trimestral em risco alto", brl(alto.receita_90d.sum()))
k4.metric("Principal causa no risco alto", TEMAS.get(alto.tema_principal.mode().iat[0], "-") if len(alto) else "-")

vista = df[df.faixa_risco.isin(faixas or ["Alto", "Médio", "Baixo"])]
if temas:
    vista = vista[vista.tema_principal.isin(temas)]
if segmentos:
    vista = vista[vista.segmento.isin(segmentos)]
vista = vista.sort_values(["prob_churn", "receita_90d"], ascending=False).reset_index(drop=True)

esq, dir_ = st.columns([3, 1.6])
with esq:
    st.subheader(f"Carteira em risco ({len(vista)})")
    st.caption("Clique em uma linha para ver o cliente.")
    tabela = vista.assign(
        risco=vista.prob_churn.map(formatar_risco),
        tema=vista.tema_principal.map(lambda t: TEMAS.get(t, t) if isinstance(t, str) else "-"))
    sel = st.dataframe(
        tabela[["cliente_id", "segmento", "cidade", "faixa_risco", "risco", "receita_90d", "tema", "motivo_1"]],
        hide_index=True, on_select="rerun", selection_mode="single-row", height=420,
        column_config={"cliente_id": st.column_config.TextColumn("Cliente", width="small"),
                       "segmento": "Segmento", "cidade": "Cidade",
                       "faixa_risco": "Faixa", "risco": "Risco",
                       "receita_90d": st.column_config.NumberColumn("Fatur. 90 dias", format="R$ %.0f"),
                       "tema": "Tema", "motivo_1": "Principal motivo"})
with dir_:
    st.subheader("Causas do risco")
    por_tema = (vista.tema_principal.map(lambda t: TEMAS.get(t, t) if isinstance(t, str) else "Sem tema")
                .value_counts().rename("clientes"))
    st.bar_chart(por_tema, horizontal=True, height=420)

linhas = sel.selection.rows if sel and sel.selection else []
if not len(vista):
    st.stop()
c = vista.iloc[linhas[0] if linhas else 0]

# ------------------------------------------------------------------------------------ cliente
st.divider()
st.subheader(f"{c.cliente_id} · {c.segmento} · {c.cidade}")
a, b, d, e = st.columns(4)
a.metric("Risco de parar de comprar (90 dias)", formatar_risco(c.prob_churn), c.faixa_risco, delta_color="off")
b.metric("Faturamento últimos 90 dias", brl(c.receita_90d))
d.metric("Última compra", f"há {int(c.recencia_dias)} dias")
e.metric("Vendedor", c.vendedor, c.canal, delta_color="off")
motivos = [c[f"motivo_{i}"] for i in (1, 2, 3) if isinstance(c.get(f"motivo_{i}"), str)]
if motivos:
    st.markdown("**Por que o modelo aponta risco (SHAP):** " + " · ".join(f"`{m}`" for m in motivos))

LIMITE_IA = int(os.environ.get("LIMITE_PLANOS_IA", "5"))
usados = st.session_state.setdefault("planos_ia", 0)
chave = f"{c.cliente_id}|{modo}|{modelo}"
bloqueado = modo == "llm" and usados >= LIMITE_IA and chave not in st.session_state
if modo == "llm":
    st.caption(f"Planos com IA nesta sessão: {usados}/{LIMITE_IA}. Depois disso, use o modo Regras.")
if st.button("✨ Gerar plano de retenção", type="primary", disabled=bloqueado):
    with st.spinner("Buscando políticas e montando o plano…"):
        try:
            st.session_state[chave] = gerar_plano(c.cliente_id, modo, base(), modelo)
            if modo == "llm":
                st.session_state["planos_ia"] += 1
        except Exception as erro:
            st.error(f"Não foi possível gerar com a IA ({erro}). Use o modo Regras.")

r = st.session_state.get(chave)
if r:
    p = r["plano"]
    aba_plano, aba_pol, aba_como = st.tabs(["📋 Plano de ação", "📚 Políticas usadas (RAG)", "⚙️ Como foi gerado"])
    with aba_plano:
        if r["validacao"]:
            for v in r["validacao"]:
                (st.error if v["nivel"] == "🔴" else st.warning)(v["mensagem"])
        else:
            st.success("Validação automática: o plano respeita as políticas (alçadas, citações e mensagem).")
        st.markdown(f"**Diagnóstico.** {p.diagnostico}")
        st.markdown(f"**Prioridade:** {p.prioridade} · **primeiro contato:** {p.prazo_primeiro_contato}")
        for i, x in enumerate(p.acoes, 1):
            st.markdown(f"**{i}. {x.descricao}**  \n"
                        f"`{x.politica}` · {x.responsavel} · {x.prazo}"
                        + (f" · 💬 {x.condicao_comercial}" if x.condicao_comercial else ""))
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Mensagem para o cliente (WhatsApp)**")
            st.code(p.mensagem_cliente, language=None, wrap_lines=True)
        with m2:
            st.markdown("**Roteiro da ligação**")
            st.markdown("\n".join(f"- {q}" for q in p.roteiro_ligacao))
            st.markdown("**Evitar**")
            st.markdown("\n".join(f"- {q}" for q in p.o_que_evitar))
    with aba_pol:
        ctx = r["contexto"]
        st.caption(f"Consulta enviada à base vetorial (tema: {ctx['tema_principal']}): {ctx['consulta']}")
        for t in ctx["politicas"]:
            with st.expander(f"[{t['doc_id']}] {t['titulo']} › {t['secao']}  ·  similaridade {t['similaridade']}"):
                st.markdown(t["texto"])
        st.caption(f"+ {len(ctx['regras_gerais'])} trechos das regras gerais (POL-10), sempre incluídos.")
    with aba_como:
        st.markdown(
            f"- **Modo:** {'LangChain + OpenAI (' + r['modelo'] + '), saída estruturada em Pydantic' if r['modo'] == 'llm' else 'regras sobre as políticas recuperadas (sem API)'}\n"
            f"- **Embeddings da busca:** {base().emb.nome}\n"
            "- **Validação:** confere se cada ação cita uma política recuperada, se descontos respeitam as alçadas "
            "da POL-01, se cliente com problema financeiro não recebe desconto (POL-02) e se a mensagem não expõe "
            "a classificação de risco (POL-10).")
        st.json(p.model_dump(), expanded=False)

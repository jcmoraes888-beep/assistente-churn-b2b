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


def plano_markdown(c, p, r) -> str:
    """Plano pronto para enviar ao vendedor (abre no Word, Notion ou e-mail)."""
    linhas = [f"# Plano de retenção · {c.cliente_id}", "",
              f"**{c.razao_social}** · {c.segmento} · {c.cidade} · vendedor {c.vendedor}",
              f"Risco: {formatar_risco(c.prob_churn)} (faixa {c.faixa_risco}) · prioridade {p.prioridade} · "
              f"primeiro contato {p.prazo_primeiro_contato}", "", f"**Diagnóstico.** {p.diagnostico}", "",
              "## Ações"]
    for i, x in enumerate(p.acoes, 1):
        linhas.append(f"{i}. {x.descricao} — *{x.responsavel}, {x.prazo}* [{x.politica}]"
                      + (f"  \n   Condição: {x.condicao_comercial}" if x.condicao_comercial else ""))
    linhas += ["", "## Mensagem para o cliente", "", p.mensagem_cliente, "", "## Roteiro da ligação"]
    linhas += [f"- {q}" for q in p.roteiro_ligacao] + ["", "## Evitar"] + [f"- {q}" for q in p.o_que_evitar]
    linhas += ["", f"_Gerado por {'IA (' + r['modelo'] + ')' if r['modo'] == 'llm' else 'regras'} · "
               "Assistente de Churn B2B · dados fictícios_"]
    return "\n".join(linhas)


# ------------------------------------------------------------------------------------ barra lateral
df = scores()
with st.sidebar:
    st.header("📉 Assistente de Churn")
    st.caption("Distribuidora Modelo Oeste · dados 100% fictícios")
    busca = st.text_input("🔎 Buscar cliente", placeholder="Código ou nome, ex.: C00700",
                          help="Procura em toda a carteira, ignorando os filtros abaixo.").strip()
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

if busca:
    alvo = busca.lower()
    vista = df[df.cliente_id.str.lower().str.contains(alvo, regex=False)
               | df.razao_social.str.lower().str.contains(alvo, regex=False)]
else:
    vista = df[df.faixa_risco.isin(faixas or ["Alto", "Médio", "Baixo"])]
    if temas:
        vista = vista[vista.tema_principal.isin(temas)]
    if segmentos:
        vista = vista[vista.segmento.isin(segmentos)]
vista = vista.sort_values(["prob_churn", "receita_90d"], ascending=False).reset_index(drop=True)

por_tema = (vista.tema_principal.map(lambda t: TEMAS.get(t, t) if isinstance(t, str) else "Sem tema")
            .value_counts().rename("clientes"))
with st.expander(f"📊 Causas do risco ({len(vista)} clientes no filtro)", expanded=False):
    st.bar_chart(por_tema, horizontal=True, height=260)

st.subheader(f"Carteira em risco ({len(vista)})" if not busca else f"Resultado da busca ({len(vista)})")
st.caption("Clique em uma linha para ver o cliente.")
tabela = vista.assign(
    risco=vista.prob_churn.map(formatar_risco),
    tema=vista.tema_principal.map(lambda t: TEMAS.get(t, t) if isinstance(t, str) else "-"))
sel = st.dataframe(
    tabela[["cliente_id", "segmento", "cidade", "faixa_risco", "risco", "receita_90d", "vendedor", "tema", "motivo_1"]],
    hide_index=True, on_select="rerun", selection_mode="single-row", height=400, width="stretch",
    column_config={"cliente_id": st.column_config.TextColumn("Cliente", width="small"),
                   "segmento": "Segmento", "cidade": "Cidade",
                   "faixa_risco": st.column_config.TextColumn("Faixa", width="small"),
                   "risco": "Risco",
                   "receita_90d": st.column_config.NumberColumn("Fatur. 90 dias", format="R$ %.0f"),
                   "vendedor": st.column_config.TextColumn("Vend.", width="small"),
                   "tema": "Tema", "motivo_1": st.column_config.TextColumn("Principal motivo", width="large")})

linhas = sel.selection.rows if sel and sel.selection else []
if not len(vista):
    st.info("Nenhum cliente encontrado. Ajuste a busca ou os filtros.")
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
        trechos = {}
        for t in r["contexto"]["politicas"] + r["contexto"]["regras_gerais"]:
            trechos.setdefault(t["doc_id"], []).append(t)
        for i, x in enumerate(p.acoes, 1):
            txt, pop = st.columns([5, 1])
            txt.markdown(f"**{i}. {x.descricao}**  \n"
                         f"`{x.politica}` · {x.responsavel} · {x.prazo}"
                         + (f" · 💬 {x.condicao_comercial}" if x.condicao_comercial else ""))
            with pop.popover(f"📖 {x.politica}", width="stretch"):
                for t in trechos.get(x.politica, []):
                    st.markdown(f"**{t['titulo']} › {t['secao']}**")
                    st.markdown(t["texto"].split("\n", 1)[-1])
                if x.politica not in trechos:
                    st.warning("Política não encontrada entre os trechos recuperados.")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Mensagem para o cliente (WhatsApp)**")
            st.code(p.mensagem_cliente, language=None, wrap_lines=True)
            st.caption("Use o ícone 📋 no canto do quadro para copiar com 1 clique.")
        with m2:
            st.markdown("**Roteiro da ligação**")
            st.markdown("\n".join(f"- {q}" for q in p.roteiro_ligacao))
            st.markdown("**Evitar**")
            st.markdown("\n".join(f"- {q}" for q in p.o_que_evitar))
        st.divider()
        md = plano_markdown(c, p, r)
        csv = pd.DataFrame([{"cliente_id": c.cliente_id, "vendedor": c.vendedor, "prioridade": p.prioridade,
                             "ordem": i, "acao": x.descricao, "responsavel": x.responsavel, "prazo": x.prazo,
                             "condicao": x.condicao_comercial or "", "politica": x.politica}
                            for i, x in enumerate(p.acoes, 1)]).to_csv(index=False, sep=";").encode("utf-8-sig")
        e1, e2, _ = st.columns([1, 1, 2])
        e1.download_button("⬇️ Plano (Markdown)", md.encode("utf-8"), f"plano_{c.cliente_id}.md",
                           "text/markdown", width="stretch")
        e2.download_button("⬇️ Ações (CSV/Excel)", csv, f"acoes_{c.cliente_id}.csv", "text/csv", width="stretch")
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

"""Monitoramento de data drift: os clientes de hoje ainda se parecem com os dados de treino?

Compara o retrato de REFERÊNCIA (conjunto de teste, períodos sem o choque) com a carteira ATUAL.
  - PSI (Population Stability Index) calculado aqui, para cada atributo: transparente e testável.
  - Relatório completo do Evidently em HTML (se a biblioteca estiver instalada).

Leitura do PSI:  < 0,10 estável · 0,10–0,25 atenção · > 0,25 drift forte

Uso:  python -m churn_b2b.monitoramento.drift
Saída: reports/drift.md, reports/drift.png e reports/drift_evidently.html
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config as C
from ..features import CATEGORICAS, NUMERICAS, construir_retrato
from ..simulacao import carregar

LIMITE_ATENCAO, LIMITE_FORTE = 0.10, 0.25
# Atributos que mudam naturalmente com o tempo: entram no relatório, mas não disparam re-treino
DRIFT_ESPERADO = {"tempo_relacionamento_meses"}


def psi_numerico(ref: pd.Series, atual: pd.Series, bins: int = 10, piso: float = 1e-4) -> float:
    cortes = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:   # atributo quase constante: compara a média
        cortes = np.unique(np.concatenate([cortes, [ref.mean() + 1e-9]]))
    cortes[0], cortes[-1] = -np.inf, np.inf
    e = np.histogram(ref.dropna(), cortes)[0] / max(len(ref.dropna()), 1)
    a = np.histogram(atual.dropna(), cortes)[0] / max(len(atual.dropna()), 1)
    e, a = np.clip(e, piso, None), np.clip(a, piso, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorico(ref: pd.Series, atual: pd.Series) -> float:
    cats = sorted(set(ref.dropna()) | set(atual.dropna()))
    e = ref.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    a = atual.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=1e-4)
    return float(np.sum((a - e) * np.log(a / e)))


def status(psi: float) -> str:
    return "🔴 drift forte" if psi > LIMITE_FORTE else ("🟠 atenção" if psi > LIMITE_ATENCAO else "🟢 estável")


def comparar(ref: pd.DataFrame, atual: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for col in NUMERICAS:
        p = psi_numerico(ref[col], atual[col])
        linhas.append({"atributo": col, "tipo": "numérico", "psi": p, "media_referencia": ref[col].mean(),
                       "media_atual": atual[col].mean()})
    for col in CATEGORICAS:
        linhas.append({"atributo": col, "tipo": "categórico", "psi": psi_categorico(ref[col], atual[col])})
    df = pd.DataFrame(linhas).sort_values("psi", ascending=False).reset_index(drop=True)
    df["status"] = df["psi"].map(status)
    return df


def drift_por_segmento(ref: pd.DataFrame, atual: pd.DataFrame, atributos: list[str],
                       segmento: str = "cidade") -> pd.DataFrame:
    """PSI dentro de cada segmento: um drift localizado (ex.: só nas cidades distantes)
    fica diluído na carteira inteira e pode passar despercebido no PSI global."""
    linhas = []
    for valor in sorted(set(ref[segmento]) & set(atual[segmento])):
        r, a = ref[ref[segmento] == valor], atual[atual[segmento] == valor]
        for col in atributos:
            # poucos clientes por segmento: menos faixas e piso maior, para o PSI não explodir por ruído
            linhas.append({segmento: valor, "atributo": col, "psi": psi_numerico(r[col], a[col], bins=5, piso=1e-2),
                           "clientes_atuais": len(a)})
    df = pd.DataFrame(linhas).sort_values("psi", ascending=False).reset_index(drop=True)
    df["status"] = df["psi"].map(status)
    return df


def _resumo_alerta(fortes: pd.DataFrame, seg_fortes: pd.DataFrame) -> str:
    partes = list(fortes.atributo)
    for atrib, g in seg_fortes.groupby("atributo"):
        partes.append(f"{atrib} em {len(g)} cidade(s): {', '.join(sorted(g.cidade))}")
    return "; ".join(partes)


def _evidently_html(ref: pd.DataFrame, atual: pd.DataFrame, destino) -> bool:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return False
    colunas = NUMERICAS + CATEGORICAS
    Report([DataDriftPreset()]).run(current_data=atual[colunas], reference_data=ref[colunas]).save_html(str(destino))
    return True


def main():
    ref = pd.read_csv(C.DADOS / "referencia_teste.csv")
    t = carregar()
    atual = construir_retrato(t["clientes"], t["pedidos"], t["atendimentos"], C.data_corte(C.CORTE_PRODUCAO),
                              com_rotulo=False)
    res = comparar(ref, atual)

    # recorte que explica o drift: entregas atrasadas por cidade
    por_cidade = pd.DataFrame({
        "referencia": ref.groupby("cidade")["pct_entregas_atrasadas_90d"].mean(),
        "atual": atual.groupby("cidade")["pct_entregas_atrasadas_90d"].mean(),
    }).sort_values("atual", ascending=False)

    suspeitos = [x for x in res[(res.tipo == "numérico") & (res.psi > LIMITE_ATENCAO)].atributo
                 if x not in DRIFT_ESPERADO] or list(res.atributo[:3])
    seg = drift_por_segmento(ref, atual, suspeitos)
    seg_fortes = seg[seg.psi > LIMITE_FORTE]
    fortes = res[(res.psi > LIMITE_FORTE) & ~res.atributo.isin(DRIFT_ESPERADO)]
    retreinar = len(fortes) > 0 or len(seg_fortes) > 0
    C.RELATORIOS.mkdir(exist_ok=True)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    top = res.head(10).iloc[::-1]
    cores = ["#d1242f" if p > LIMITE_FORTE else ("#bf8700" if p > LIMITE_ATENCAO else "#1a7f37") for p in top.psi]
    ax.barh(top.atributo, top.psi, color=cores)
    ax.axvline(LIMITE_ATENCAO, ls="--", c="#bf8700", lw=1, label="atenção (0,10)")
    ax.axvline(LIMITE_FORTE, ls="--", c="#d1242f", lw=1, label="drift forte (0,25)")
    ax.set_xlim(0, max(0.3, res.psi.max() * 1.1))
    ax.set_xlabel("PSI")
    n_glob = int((res.psi > LIMITE_FORTE).sum())
    ax.set_title("Carteira inteira: nenhum atributo passa de 0,25" if n_glob == 0
                 else f"Carteira inteira: {n_glob} atributo(s) acima de 0,25")
    ax.legend(fontsize=8, loc="lower right")

    y = np.arange(len(por_cidade))
    ax2.barh(y + 0.2, por_cidade.referencia * 100, height=0.4, color="#8c959f", label="referência")
    ax2.barh(y - 0.2, por_cidade.atual * 100, height=0.4, color="#d1242f", label="atual")
    ax2.set_yticks(y, por_cidade.index)
    ax2.invert_yaxis()
    ax2.set_xlabel("% das entregas com atraso (90 dias)")
    ax2.set_title("Entregas com atraso por cidade")
    ax2.legend(fontsize=8, loc="lower right")
    for a_ in (ax, ax2):
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Monitoramento de data drift · referência (teste) × carteira atual", fontsize=11, color="#57606a")
    fig.tight_layout()
    fig.savefig(C.RELATORIOS / "drift.png", dpi=130)
    plt.close(fig)

    tabela = res.assign(psi=res.psi.round(3), media_referencia=res.media_referencia.round(3),
                        media_atual=res.media_atual.round(3)).head(12)
    pc = (por_cidade * 100).round(1)
    md = ["# Monitoramento de data drift", "",
          f"Referência: conjunto de teste ({len(ref):,} retratos) · Atual: carteira em "
          f"{C.data_corte(C.CORTE_PRODUCAO).date()} ({len(atual):,} clientes).".replace(",", "."), "",
          f"**Atributos com drift forte (PSI > {LIMITE_FORTE}): {len(fortes)}** · "
          f"atenção: {int(((res.psi > LIMITE_ATENCAO) & (res.psi <= LIMITE_FORTE)).sum())}", "",
          ("> ⚠️ **Recomendação: investigar e re-treinar o modelo.** Drift forte em "
           + _resumo_alerta(fortes, seg_fortes) + ".") if retreinar else "> ✅ Sem drift forte: o modelo pode seguir em produção.", "",
          "## Visão global (carteira inteira)", "",
          tabela.to_markdown(index=False), "",
          "## Visão por cidade (atributos em atenção)", "",
          "O PSI global pode esconder um problema localizado. Por cidade (5 faixas, por causa da amostra menor):", "",
          seg.head(10).assign(psi=seg.psi.round(3)).to_markdown(index=False), "",
          "## Entregas atrasadas por cidade (% dos pedidos, últimos 90 dias)", "",
          pc.rename(columns={"referencia": "referência (%)", "atual": "atual (%)"}).to_markdown(), "",
          "![Drift](drift.png)"]
    (C.RELATORIOS / "drift.md").write_text("\n".join(md), "utf-8")
    (C.RELATORIOS / "drift.json").write_text(json.dumps(
        {"retreinar": retreinar, "atributos_drift_forte": list(fortes.atributo),
         "drift_forte_por_cidade": [f"{r.atributo} ({r.cidade})" for r in seg_fortes.itertuples()],
         "psi": dict(zip(res.atributo, res.psi.round(4)))}, ensure_ascii=False, indent=2), "utf-8")
    html = _evidently_html(ref, atual, C.RELATORIOS / "drift_evidently.html")

    print(res.head(8)[["atributo", "psi", "status"]].to_string(index=False))
    print("\nEntregas atrasadas por cidade (%):\n" + pc.to_string())
    print("\nPSI por cidade (maiores):\n" + seg.head(6).assign(psi=seg.psi.round(3)).to_string(index=False))
    print("\n" + ("⚠️  Drift forte em " + _resumo_alerta(fortes, seg_fortes) + " → recomendação: re-treinar"
                  if retreinar else "✅ Sem drift forte"))
    print("Relatórios: reports/drift.md, reports/drift.png" + (", reports/drift_evidently.html" if html else ""))


if __name__ == "__main__":
    main()

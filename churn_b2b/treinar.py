"""Treina e compara os modelos, registra tudo no MLflow e salva o campeão em models/.

Uso:  python -m churn_b2b.treinar
Ver os experimentos:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import json
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score)

from . import config as C
from .features import ATRIBUTOS, construir_varios
from .modelos import ModeloChurn, RedeNeuralChurn, candidatos
from .simulacao import carregar

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def metricas(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    corte10 = np.quantile(p, 0.9)
    top = p >= corte10
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "lift_top10": y[top].mean() / y.mean(),
        "recall_top10": y[top].sum() / y.sum(),
        "taxa_churn": y.mean(),
    }


def _mlflow():
    try:
        import mlflow
        mlflow.set_tracking_uri(C.MLFLOW_URI)
        mlflow.set_experiment(C.MLFLOW_EXPERIMENTO)
        return mlflow
    except Exception as e:  # MLflow é opcional para rodar o treino
        print(f"[aviso] MLflow indisponível ({e}); seguindo sem registro.")
        return None


def main():
    t0 = time.time()
    tabelas = carregar()
    cortes = lambda ms: [C.data_corte(m) for m in ms]  # noqa: E731
    treino = construir_varios(tabelas, cortes(C.CORTES_TREINO))
    valid = construir_varios(tabelas, cortes(C.CORTES_VALIDACAO))
    teste = construir_varios(tabelas, cortes(C.CORTES_TESTE))
    print(f"Treino {len(treino):,} ({treino.churn.mean():.1%} churn) · "
          f"Validação {len(valid):,} ({valid.churn.mean():.1%}) · Teste {len(teste):,} ({teste.churn.mean():.1%})")

    mlflow = _mlflow()
    resultados = []
    for familia, opcoes in candidatos().items():
        for params, est in opcoes:
            m = ModeloChurn(familia, est).fit(treino[ATRIBUTOS], treino.churn, valid[ATRIBUTOS], valid.churn)
            met = metricas(valid.churn, m.predict_proba(valid[ATRIBUTOS]))
            resultados.append({"familia": familia, "params": params, "modelo": m, **met})
            print(f"  {familia:<20} {str(params):<55} val PR-AUC {met['pr_auc']:.3f}  ROC-AUC {met['roc_auc']:.3f}")
            if mlflow:
                with mlflow.start_run(run_name=familia):
                    mlflow.set_tag("familia", familia)
                    mlflow.log_params({k: str(v) for k, v in params.items()})
                    mlflow.log_metrics({f"val_{k}": v for k, v in met.items()})
                    if isinstance(est, RedeNeuralChurn):
                        for h in est.historico:
                            mlflow.log_metric("train_loss", h["loss"], step=h["epoca"])
                            mlflow.log_metric("val_pr_auc_epoca", h["val_pr_auc"], step=h["epoca"])

    res = pd.DataFrame(resultados)
    melhores = res.loc[res.groupby("familia")["pr_auc"].idxmax()].sort_values("pr_auc", ascending=False)
    campeao = melhores.iloc[0]

    for m in melhores.modelo:  # calibra as probabilidades dos finalistas na validação
        m.calibrar(valid[ATRIBUTOS], valid.churn)

    # avaliação final no TESTE (períodos que o modelo nunca viu)
    linhas, curvas = [], {}
    for _, r in melhores.iterrows():
        p_teste = r.modelo.predict_proba(teste[ATRIBUTOS])
        mt = metricas(teste.churn, p_teste)
        curvas[r.familia] = precision_recall_curve(teste.churn, p_teste)
        linhas.append({"modelo": r.familia, "params": str(r.params),
                       **{f"val_{k}": r[k] for k in ["pr_auc", "roc_auc"]},
                       **{f"teste_{k}": v for k, v in mt.items()}})
    comparativo = pd.DataFrame(linhas)

    # faixas de risco calibradas na validação: top 10% = Alto, próximos 20% = Médio
    p_val = campeao.modelo.predict_proba(valid[ATRIBUTOS])
    faixas = {"alto": float(np.quantile(p_val, 0.90)), "medio": float(np.quantile(p_val, 0.70))}

    C.MODELOS.mkdir(exist_ok=True)
    C.RELATORIOS.mkdir(exist_ok=True)
    joblib.dump(campeao.modelo, C.MODELOS / "modelo_churn.joblib")
    meta = {
        "modelo": campeao.familia, "params": {k: str(v) for k, v in campeao.params.items()},
        "atributos": ATRIBUTOS, "faixas_risco": faixas,
        "metricas_teste": comparativo.set_index("modelo").loc[campeao.familia].filter(like="teste_").to_dict(),
        "treinado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
        "cortes": {"treino": [str(c.date()) for c in cortes(C.CORTES_TREINO)],
                   "validacao": [str(c.date()) for c in cortes(C.CORTES_VALIDACAO)],
                   "teste": [str(c.date()) for c in cortes(C.CORTES_TESTE)]},
    }
    (C.MODELOS / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    teste.to_csv(C.DADOS / "referencia_teste.csv", index=False)  # base de referência para o drift

    # relatório
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for nome, (prec, rec, _) in curvas.items():
        ax.plot(rec, prec, label=f"{nome} (PR-AUC {average_precision_score(teste.churn, melhores.set_index('familia').loc[nome].modelo.predict_proba(teste[ATRIBUTOS])):.3f})")
    ax.axhline(teste.churn.mean(), ls="--", c="gray", lw=1, label=f"aleatório ({teste.churn.mean():.1%})")
    ax.set_xlabel("Recall (churns encontrados)")
    ax.set_ylabel("Precisão (acerto nos alertas)")
    ax.set_title("Curva precisão × recall — conjunto de teste")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.RELATORIOS / "curva_pr_teste.png", dpi=130)

    tabela = comparativo[["modelo", "val_pr_auc", "teste_pr_auc", "teste_roc_auc", "teste_lift_top10",
                          "teste_recall_top10", "teste_brier"]].round(3)
    md = ["# Resultado do treino", "",
          f"Modelo escolhido: **{campeao.familia}** (maior PR-AUC na validação).", "",
          f"Taxa de churn no teste: {teste.churn.mean():.1%}. "
          "PR-AUC é a métrica principal porque a classe é rara.", "",
          tabela.to_markdown(index=False), "",
          "- **lift_top10**: quantas vezes o top 10% de risco concentra mais churn que a média.",
          "- **recall_top10**: % dos churns encontrados olhando só os 10% de maior risco.", "",
          "![Curva PR](curva_pr_teste.png)"]
    (C.RELATORIOS / "resultado_treino.md").write_text("\n".join(md), "utf-8")

    if mlflow:
        with mlflow.start_run(run_name=f"campeao_{campeao.familia}"):
            mlflow.set_tag("campeao", "sim")
            mlflow.log_params(meta["params"])
            mlflow.log_metrics({k: float(v) for k, v in meta["metricas_teste"].items()})
            for arq in ["modelo_churn.joblib", "metadata.json"]:
                mlflow.log_artifact(str(C.MODELOS / arq))
            mlflow.log_artifact(str(C.RELATORIOS / "curva_pr_teste.png"))
    print("\n" + tabela.to_string(index=False))
    print(f"\nCampeão: {campeao.familia} · salvo em models/ · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

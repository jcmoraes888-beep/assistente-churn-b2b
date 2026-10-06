"""Gera os dados brutos de uma distribuidora de alimentos FICTÍCIA do oeste do Paraná.

Tabelas geradas em data/:
  clientes.csv      cadastro (segmento, cidade, canal, vendedor, prazo, início do relacionamento)
  pedidos.csv       um registro por pedido (valor, itens, desconto, atraso de entrega e de pagamento)
  atendimentos.csv  chamados/reclamações abertos pelo cliente
  verdade.csv       gabarito da simulação (mês e causa real do churn) — NUNCA usar como feature

Lógica da simulação
  1. Cada cliente tem fatores ocultos (sensibilidade a preço, risco financeiro, distância).
  2. Um risco mensal (hazard) define se e quando o cliente deixa de comprar.
  3. Cada churn tem uma causa: serviço, preço, financeiro ou "silencioso".
  4. Nos meses antes do churn aparecem sinais coerentes com a causa
     (menos pedidos, mix menor, mais atrasos, reclamações, pedidos de desconto...).
     Os churns "silenciosos" quase não deixam sinal — para o modelo não ficar perfeito demais.
  5. Nos últimos meses há um choque logístico nas cidades distantes (data drift proposital,
     usado na etapa de monitoramento com Evidently).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DADOS, INICIO, MESES, N_CLIENTES, SEED

# segmento: (ticket médio R$, pedidos por mês, peso na base, efeito no risco)
SEGMENTOS = {
    "Supermercado": (2800, 6.0, 0.18, -0.2),
    "Mercearia": (900, 4.0, 0.25, 0.1),
    "Padaria": (1100, 5.0, 0.20, 0.0),
    "Restaurante": (1500, 4.0, 0.17, 0.25),
    "Lanchonete": (600, 3.0, 0.12, 0.45),
    "Hotel": (2000, 3.0, 0.08, -0.1),
}
# cidade: (km do centro de distribuição, peso)
CIDADES = {
    "Cascavel": (0, 0.38), "Toledo": (50, 0.18), "Foz do Iguaçu": (140, 0.16),
    "Marechal Cândido Rondon": (85, 0.10), "Medianeira": (90, 0.09), "Palotina": (95, 0.09),
}
CANAIS = {"Vendedor externo": (0.55, -0.15), "Televendas": (0.30, 0.10), "E-commerce B2B": (0.15, 0.25)}
VENDEDORES = {f"V{i:02d}": e for i, e in enumerate([-0.3, -0.1, 0.0, 0.0, 0.1, 0.2, 0.35, 0.5], 1)}
PRAZOS = [7, 14, 21, 28]

TIPOS_ATENDIMENTO = ["Atraso na entrega", "Qualidade do produto", "Preço/condição comercial",
                     "Cobrança/financeiro", "Falta de produto"]
MESES_CHOQUE = range(MESES - 3, MESES)   # choque logístico nos 3 últimos meses


def _escolher(rng, opcoes: dict, idx_peso: int, n: int):
    nomes = list(opcoes)
    pesos = np.array([opcoes[k][idx_peso] for k in nomes], dtype=float)
    return rng.choice(nomes, size=n, p=pesos / pesos.sum())


def gerar(n_clientes: int = N_CLIENTES, seed: int = SEED, salvar: bool = True) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------- cadastro
    seg = _escolher(rng, SEGMENTOS, 2, n_clientes)
    cid = _escolher(rng, CIDADES, 1, n_clientes)
    canal = _escolher(rng, CANAIS, 0, n_clientes)
    vend = rng.choice(list(VENDEDORES), size=n_clientes)
    prazo = rng.choice(PRAZOS, size=n_clientes, p=[0.2, 0.35, 0.25, 0.2])

    # início: 60% já eram clientes antes da simulação; 40% entram ao longo do tempo
    antigo = rng.random(n_clientes) < 0.6
    mes_inicio = np.where(antigo, 0, rng.integers(1, MESES - 4, n_clientes))
    meses_antes = np.where(antigo, rng.integers(3, 72, n_clientes), 0)
    inicio = [INICIO + pd.DateOffset(months=int(m)) - pd.DateOffset(months=int(a))
              + pd.Timedelta(days=int(rng.integers(0, 28)))
              for m, a in zip(mes_inicio, meses_antes)]

    porte = rng.lognormal(0, 0.45, n_clientes)                # tamanho relativo do cliente
    sens_preco = rng.beta(2, 3, n_clientes)                  # 0..1
    risco_fin = rng.beta(1.5, 5, n_clientes)                 # 0..1
    dist = np.array([CIDADES[c][0] for c in cid])

    clientes = pd.DataFrame({
        "cliente_id": [f"C{i:05d}" for i in range(1, n_clientes + 1)],
        "razao_social": [f"Cliente Fictício {i:05d}" for i in range(1, n_clientes + 1)],
        "segmento": seg, "cidade": cid, "canal": canal, "vendedor": vend,
        "prazo_pagamento_dias": prazo, "inicio_relacionamento": pd.to_datetime(inicio).normalize(),
    })

    # ------------------------------------------------ quando cada um sai (hazard)
    mes_churn = np.full(n_clientes, -1)
    for i in range(n_clientes):
        for m in range(mes_inicio[i], MESES):
            meses_cliente = meses_antes[i] + (m - mes_inicio[i])
            logit = (-5.2 + 1.6 * sens_preco[i] + 1.4 * risco_fin[i] + 0.006 * dist[i]
                     + SEGMENTOS[seg[i]][3] + CANAIS[canal[i]][1] + VENDEDORES[vend[i]]
                     + (0.8 if meses_cliente < 6 else 0.0) - 0.25 * np.log(porte[i] + 0.5)
                     + (1.6 * dist[i] / 140 if m in MESES_CHOQUE else 0.0))
            if rng.random() < 1 / (1 + np.exp(-logit)):
                mes_churn[i] = m
                break

    # causa do churn, ligada aos fatores ocultos
    causa = np.array([""] * n_clientes, dtype=object)
    for i in np.where(mes_churn >= 0)[0]:
        pesos = np.array([0.6 + dist[i] / 100, 0.4 + 1.5 * sens_preco[i], 0.2 + 2.0 * risco_fin[i], 0.9])
        causa[i] = rng.choice(["servico", "preco", "financeiro", "silencioso"], p=pesos / pesos.sum())

    # ------------------------------------------------ pedidos e atendimentos
    pedidos, atend = [], []
    quedas_temporarias = {}
    for i in np.where(rng.random(n_clientes) < 0.25)[0]:
        ini = int(rng.integers(mes_inicio[i], MESES))
        quedas_temporarias[i] = set(range(ini, ini + int(rng.integers(2, 4))))
    for i in range(n_clientes):
        cid_i = clientes.cliente_id.iat[i]
        ticket, freq = SEGMENTOS[seg[i]][0] * porte[i], SEGMENTOS[seg[i]][1] * min(porte[i], 2.0) ** 0.5
        fim = mes_churn[i] if mes_churn[i] >= 0 else MESES
        for m in range(mes_inicio[i], fim):
            faltam = (mes_churn[i] - m) if mes_churn[i] >= 0 else 99
            sinal = causa[i] != "silencioso" and faltam <= 4
            queda = {1: 0.5, 2: 0.7, 3: 0.85, 4: 0.95}.get(faltam, 1.0) if sinal else \
                ({1: 0.85, 2: 0.95}.get(faltam, 1.0))
            # "falsos alarmes": clientes que esfriam por um tempo (obra, férias, sazonalidade) e voltam
            if m in quedas_temporarias.get(i, ()):
                queda *= 0.45
            sazonal = 1.2 if (INICIO.month + m - 1) % 12 + 1 == 12 else (0.9 if (INICIO.month + m - 1) % 12 + 1 == 1 else 1.0)
            n_ped = rng.poisson(freq * queda * sazonal)

            p_atraso = 0.04 + dist[i] / 1400 + (0.22 * dist[i] / 140 if m in MESES_CHOQUE else 0.0)
            if sinal and causa[i] == "servico":
                p_atraso += 0.15
            ini_mes = INICIO + pd.DateOffset(months=m)
            dias_mes = ((ini_mes + pd.DateOffset(months=1)) - ini_mes).days
            for _ in range(n_ped):
                atrasou = rng.random() < p_atraso
                desc = max(0.0, rng.normal(2.5 + 3 * sens_preco[i], 1.0))
                if sinal and causa[i] == "preco":
                    desc += rng.uniform(0.5, 4)
                atraso_pgto = max(0, int(rng.normal(12 * risco_fin[i] - 2, 3)))
                if sinal and causa[i] == "financeiro":
                    atraso_pgto += int(rng.integers(3, 20))
                pedidos.append((
                    cid_i, ini_mes + pd.Timedelta(days=int(rng.integers(0, dias_mes))),
                    round(ticket * rng.lognormal(0, 0.35) * (0.75 + 0.25 * queda), 2),
                    int(rng.poisson(9 * min(porte[i], 2.5) ** 0.5 * (0.6 + 0.4 * queda)) + 1),
                    round(desc, 2), int(1 + rng.geometric(0.5)) if atrasou else 0, atraso_pgto,
                ))
                if atrasou and rng.random() < 0.35:
                    atend.append((cid_i, pedidos[-1][1] + pd.Timedelta(days=2), "Atraso na entrega"))

            # chamados "de fundo" + chamados ligados à causa do churn
            for _ in range(rng.poisson(0.2)):
                atend.append((cid_i, ini_mes + pd.Timedelta(days=int(rng.integers(0, dias_mes))),
                              rng.choice(TIPOS_ATENDIMENTO, p=[0.1, 0.35, 0.2, 0.1, 0.25])))
            if sinal:
                tipo = {"servico": "Atraso na entrega", "preco": "Preço/condição comercial",
                        "financeiro": "Cobrança/financeiro"}[causa[i]]
                for _ in range(rng.poisson(0.45)):
                    atend.append((cid_i, ini_mes + pd.Timedelta(days=int(rng.integers(0, dias_mes))), tipo))

    pedidos = pd.DataFrame(pedidos, columns=["cliente_id", "data", "valor", "itens", "desconto_pct",
                                             "atraso_entrega_dias", "atraso_pagamento_dias"])
    pedidos = pedidos.sort_values(["data", "cliente_id"]).reset_index(drop=True)
    pedidos.insert(0, "pedido_id", [f"P{i:07d}" for i in range(1, len(pedidos) + 1)])
    atend = pd.DataFrame(atend, columns=["cliente_id", "data", "tipo"]).sort_values("data").reset_index(drop=True)
    atend = atend[atend["data"] < INICIO + pd.DateOffset(months=MESES)]

    verdade = pd.DataFrame({
        "cliente_id": clientes.cliente_id,
        "mes_churn": mes_churn,
        "data_churn": [INICIO + pd.DateOffset(months=int(m)) if m >= 0 else pd.NaT for m in mes_churn],
        "causa_churn": causa,
    })

    tabelas = {"clientes": clientes, "pedidos": pedidos, "atendimentos": atend, "verdade": verdade}
    if salvar:
        DADOS.mkdir(exist_ok=True)
        for nome, df in tabelas.items():
            df.to_csv(DADOS / f"{nome}.csv", index=False)
    return tabelas


def carregar() -> dict[str, pd.DataFrame]:
    if not (DADOS / "pedidos.csv").exists():
        return gerar()
    datas = {"clientes": ["inicio_relacionamento"], "pedidos": ["data"], "atendimentos": ["data"],
             "verdade": ["data_churn"]}
    return {n: pd.read_csv(DADOS / f"{n}.csv", parse_dates=d) for n, d in datas.items()}


if __name__ == "__main__":
    t = gerar()
    v = t["verdade"]
    print(f"Clientes: {len(t['clientes']):,} · Pedidos: {len(t['pedidos']):,} · "
          f"Atendimentos: {len(t['atendimentos']):,}")
    print(f"Churn no período: {(v.mes_churn >= 0).mean():.1%}")
    print(v[v.mes_churn >= 0].causa_churn.value_counts().to_string())

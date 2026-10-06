"""Testes das regras que mais quebram projetos de churn: vazamento de dados e definição do rótulo."""
import pandas as pd
import pytest

from churn_b2b.features import ATRIBUTOS, construir_retrato
from churn_b2b.simulacao import gerar

CORTE = pd.Timestamp("2025-06-01")


@pytest.fixture(scope="module")
def tabelas():
    return gerar(n_clientes=150, seed=7, salvar=False)


def test_sem_vazamento_do_futuro(tabelas):
    """Alterar qualquer dado POSTERIOR ao corte não pode mudar os atributos."""
    base = construir_retrato(tabelas["clientes"], tabelas["pedidos"], tabelas["atendimentos"], CORTE)
    ped = tabelas["pedidos"].copy()
    futuro = ped["data"] >= CORTE
    ped.loc[futuro, ["valor", "itens", "atraso_entrega_dias"]] *= 10
    at = tabelas["atendimentos"].copy()
    at.loc[at["data"] >= CORTE, "tipo"] = "Atraso na entrega"
    mexido = construir_retrato(tabelas["clientes"], ped, at, CORTE)
    pd.testing.assert_frame_equal(base[ATRIBUTOS], mexido[ATRIBUTOS])


def test_rotulo_churn(tabelas):
    """Churn = 1 exatamente quando o cliente não compra nos 90 dias após o corte."""
    r = construir_retrato(tabelas["clientes"], tabelas["pedidos"], tabelas["atendimentos"], CORTE)
    ped = tabelas["pedidos"]
    janela = ped[(ped["data"] >= CORTE) & (ped["data"] < CORTE + pd.Timedelta(days=90))]
    compraram = set(janela["cliente_id"])
    esperado = (~r["cliente_id"].isin(compraram)).astype(int)
    assert (r["churn"] == esperado).all()


def test_so_clientes_ativos(tabelas):
    """Só entra no retrato quem comprou nos 90 dias antes do corte."""
    r = construir_retrato(tabelas["clientes"], tabelas["pedidos"], tabelas["atendimentos"], CORTE)
    assert (r["recencia_dias"] <= 90).all()
    assert r[ATRIBUTOS].isna().sum().sum() == 0

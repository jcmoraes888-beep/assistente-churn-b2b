# Assistente de Churn B2B

[![CI](https://github.com/jcmoraes888-beep/assistente-churn-b2b/actions/workflows/ci.yml/badge.svg)](https://github.com/jcmoraes888-beep/assistente-churn-b2b/actions/workflows/ci.yml)

**[Ver a tela funcionando (demo ao vivo)](https://churn-b2b-jcmoraes.streamlit.app)**

Prevê quais clientes de uma distribuidora vão **parar de comprar**, explica **por quê**, busca nas políticas da empresa **o que pode ser feito** (RAG) e gera o **plano de ação** com LLM, conferido por regras automáticas. Tudo testado e empacotado com CI/CD, Docker e monitoramento de drift (MLOps).

> Dados **100% fictícios**, simulados para uma distribuidora de alimentos do oeste do Paraná (2.500 clientes, ~190 mil pedidos, 30 meses).

```
[Pedidos, atendimentos, cadastro] ─> Atributos por cliente ─> Modelo de churn ─> SHAP (motivos)
                                                                             │
[Políticas de retenção] ─> RAG (ChromaDB) ──────────────────────────────────┼─> LLM: plano de ação
                                                                             │
                                FastAPI + Docker + MLflow + GitHub Actions + Evidently (drift)
```

## Status

| Etapa | Conteúdo | Status |
|---|---|---|
| 1 | Dados, atributos, 3 modelos comparados, MLflow, SHAP | ✅ concluída |
| 2 | RAG: 10 políticas de retenção, ChromaDB, busca por tema e avaliação | ✅ concluída |
| 3 | Agente LLM (LangChain + OpenAI), validação automática, API FastAPI e tela Streamlit | ✅ concluída |
| 4 | MLOps: GitHub Actions (testes + Docker), imagem no GHCR, deploy da tela, monitoramento de drift | ✅ concluída |

## O problema: churn em B2B não tem "cancelamento"

Cliente de distribuidora não cancela contrato: ele **vai comprando menos até sumir**. Por isso a definição é:

> **Churn** = cliente ativo (comprou nos últimos 90 dias) que **não compra nos 90 dias seguintes**.

O modelo olha o retrato do cliente numa data e responde: *qual a chance de ele parar de comprar no próximo trimestre?*

## Etapa 1: resultados

Divisão **temporal** (treino em 2024-25, validação e teste em períodos posteriores que o modelo nunca viu), para simular o uso real.

| Modelo | PR-AUC teste | ROC-AUC teste | Lift top 10% | Recall top 10% |
|---|---|---|---|---|
| Rede neural (PyTorch) | 0,725 | 0,926 | 7,1× | 71% |
| LightGBM | 0,737 | 0,926 | 7,5× | 75% |
| Regressão logística | 0,709 | 0,925 | 7,1× | 71% |

**O que isso quer dizer na prática:** olhando só os 10% de clientes com maior risco, a equipe comercial encontra cerca de **3 em cada 4 clientes que vão sair**.

**Decisões e aprendizados**

- **PR-AUC como métrica principal**: só ~8% dos clientes saem por trimestre, e a acurácia enganaria.
- **A rede neural venceu na validação, mas o LightGBM foi melhor no teste.** Os três ficaram próximos, um resultado comum em dados tabulares. A escolha é feita pela validação, para não "espiar" o teste, e a comparação completa fica registrada no MLflow.
- **Calibração (Platt)**: o treino usa pesos para a classe rara, o que infla as probabilidades. Depois de calibrar, "risco de 30%" significa ~30% de chance real (Brier 0,041).
- **Sem vazamento**: há um teste automatizado que altera dados futuros e garante que os atributos não mudam.
- **Explicações conferidas**: como os dados são simulados, existe o gabarito da causa real. Em **87%** dos churns de teste com causa conhecida (entrega, preço ou financeiro), a causa real aparece entre os 3 motivos que o SHAP aponta.

![Curva PR](reports/curva_pr_teste.png)

### Treino × validação: detectando overfitting

![Treino x validação](reports/treino_vs_validacao.png)

Gráfico gerado automaticamente a cada treino (`python -m churn_b2b.treinar`). A configuração em **vermelho** continua baixando o erro de treino, mas a PR-AUC de validação despenca logo depois do pico: é **overfitting**. O early stopping interrompe o treino e recupera os pesos da melhor época (●). A configuração em **verde** teve a melhor validação e foi a escolhida entre as redes neurais.

> Os números podem variar um pouco entre computadores (inicialização aleatória e versões das bibliotecas), mas o padrão se repete.

### Exemplo de saída (`data/scores_atuais.csv`)

| Cliente | Risco | Motivos |
|---|---|---|
| C00700 · Mercearia · Palotina | Alto | Atraso médio de pagamento de 16 dias · Última compra há 69 dias |
| C01699 · Hotel · Cascavel | Alto | Última compra há 81 dias · Atraso médio de pagamento de 19 dias · Só 1 pedido em 90 dias |

Cada motivo recebe um **tema** (entrega, preço, financeiro, produto, engajamento, relacionamento, canal), que na Etapa 2 vai buscar a política de retenção certa no RAG.

## Etapa 2: RAG com as políticas de retenção

O modelo diz **quem** está em risco e **por quê**. O RAG responde **o que a empresa permite fazer** para reter esse cliente, buscando o trecho certo nas políticas internas.

**Base de conhecimento** (`politicas/`): 10 políticas fictícias da distribuidora, escritas com regras concretas (alçadas de desconto, prazos de entrega por cidade, renegociação de dívida, cadência de reativação, ofertas por segmento etc.). Cada política tem no cabeçalho os **temas** e os **segmentos** a que se aplica.

**Como funciona**

1. Cada política é fatiada por seção (41 trechos), vira vetor (embeddings) e vai para o **ChromaDB**.
2. Para cada cliente em risco, a consulta é montada com o perfil e os **motivos do SHAP**.
3. A busca é **semântica + filtro por metadados**: só entram políticas do **tema principal** do cliente (ex.: financeiro) e válidas para o **segmento** dele (ex.: restaurante). Um segundo tema também é consultado.
4. As **regras gerais** (prazos por faixa de risco, o que nunca fazer) vão sempre junto.

```
C00700 · Mercearia · Palotina · risco Alto
  • Atraso médio de pagamento de 16 dias
  • Última compra há 69 dias
Tema principal: financeiro
[POL-02] Crédito, cobrança e renegociação de prazo › Quando usar
[POL-02] Crédito, cobrança e renegociação de prazo › Opções de renegociação
[POL-02] Crédito, cobrança e renegociação de prazo › Primeiro contato
[POL-04] Programa de reativação de clientes que estão comprando menos › Quando usar
```

**Avaliação do recuperador**: 22 perguntas com a política correta definida à mão (`politicas/perguntas_avaliacao.json`).

| Embeddings | Filtro por tema | Acerto@1 | Acerto@3 | MRR |
|---|---|---|---|---|
| Local (n-gramas, sem API) | não | 73% | 95% | 0,84 |
| Local (n-gramas, sem API) | sim | 95% | 100% | 0,98 |
| OpenAI `text-embedding-3-small` | sim | *rode `python -m churn_b2b.rag.avaliar --comparar`* | | |

**Aprendizado:** o filtro por metadados (tema vindo do SHAP) pesa mais do que o modelo de embedding. É ele que leva o acerto@1 de 73% para 95%, porque o modelo de churn já diz *qual* é o problema e a busca só precisa achar a regra certa dentro daquele assunto.

## Etapa 3: agente de retenção (LLM) + API + tela

O agente recebe o pacote da Etapa 2 (cliente, risco, motivos do SHAP e políticas recuperadas) e devolve um **plano de retenção estruturado**: diagnóstico, prioridade, ações com responsável, prazo e **política que autoriza**, roteiro de ligação e mensagem de WhatsApp pronta para o vendedor.

**Como funciona**

- **LangChain + OpenAI com saída estruturada** (`with_structured_output` + Pydantic): o LLM é obrigado a responder no formato do `PlanoRetencao`, não em texto livre.
- **Prompt com regras de negócio**: usar só as políticas fornecidas, citar o id, respeitar alçadas, resolver a causa antes de dar desconto e nunca expor ao cliente que ele foi classificado como risco.
- **Validação automática (guardrails)** depois do LLM, sem IA:
  - toda ação cita uma política que foi de fato recuperada (detecta invenção);
  - descontos respeitam as alçadas da POL-01 (vendedor 3%, supervisor 6%, gerente 10%);
  - cliente com problema financeiro não recebe desconto antes de regularizar (POL-02);
  - a mensagem ao cliente não fala em risco, churn ou modelo (POL-10);
  - prioridade coerente com a faixa de risco.
- **Modo regras**: o mesmo pipeline monta o plano sem API. Serve para demonstração sem custo, para os testes automáticos e como plano B se a API falhar.
- **Probabilidade honesta**: valores acima de 95% aparecem como "mais de 95%", em vez de um "100%" com falsa precisão.

**Exemplo** (`python -m churn_b2b.agente.plano C00700 --modo regras`)

```
C00700 · Mercearia · Palotina · risco Alto
1. [POL-02] Contato do financeiro junto com o vendedor para propor reprogramação do vencimento
   — financeiro, em até 2 dias úteis · Condição: reprogramação em até 15 dias, sem juros
2. [POL-04] Ligação do vendedor perguntando abertamente se houve algum problema — em até 48 horas
3. [POL-04] Se não houver pedido em 30 dias, frete grátis nos 2 próximos pedidos

Mensagem: "Olá! Tudo bem? Aqui é [Nome do vendedor], da Distribuidora Modelo Oeste. Vi que os últimos
boletos ficaram um pouco apertados e quero entender como podemos ajudar..."
Validação automática: ✅ nenhum problema encontrado
```

**API (FastAPI)**: `uvicorn churn_b2b.api:app --reload` → documentação em http://localhost:8000/docs

| Método | Rota | O que faz |
|---|---|---|
| GET | `/saude` | status, nº de clientes, embeddings em uso |
| GET | `/carteira?faixa=Alto&tema=entrega` | clientes em risco, ordenados por probabilidade |
| GET | `/clientes/{id}` | risco, faixa e motivos |
| GET | `/clientes/{id}/politicas` | políticas recuperadas pelo RAG |
| POST | `/clientes/{id}/plano?modo=llm` | plano de retenção + validação |

**Tela (Streamlit)**: `streamlit run app/streamlit_app.py` mostra a carteira em risco, as causas, o cliente selecionado e o plano gerado, com as políticas usadas e a validação. Sem chave da OpenAI, funciona no modo regras. A pasta `app/` tem um `requirements.txt` próprio, sem PyTorch, para publicar no Streamlit Cloud.

## Etapa 4: MLOps

**Integração contínua (GitHub Actions)**: a cada push, `.github/workflows/ci.yml`:

1. instala as dependências e roda os **14 testes** (vazamento de dados, rótulo, modelo, RAG, agente, validação, API e drift), sem chamar a OpenAI;
2. constrói a **imagem Docker** da API e testa a API rodando dentro do container;
3. na branch `main`, publica a imagem no **GitHub Container Registry** (`ghcr.io/jcmoraes888-beep/assistente-churn-b2b`).

**Docker**: imagem leve só com o necessário para servir (RAG + agente + API, sem PyTorch), rodando com usuário sem privilégios e com healthcheck.

```bash
docker build -t churn-b2b-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... churn-b2b-api   # sem a chave: embeddings locais e modo regras
```

**Monitoramento de data drift** (`python -m churn_b2b.monitoramento.drift`): compara a carteira atual com o conjunto de referência usando **PSI** (calculado no próprio código, testável) e gera também o relatório completo do **Evidently**.

![Drift](reports/drift.png)

O caso é proposital: a simulação tem um choque logístico nas cidades distantes nos últimos meses.

- **Na carteira inteira, nenhum atributo passa do limite de drift forte** (PSI 0,25): o atraso nas entregas aparece só como "atenção".
- **Por cidade, o problema fica claro**: as entregas atrasadas dobraram em Foz do Iguaçu, Palotina, Medianeira, Marechal Cândido Rondon e Toledo (PSI > 1), enquanto Cascavel, onde fica o centro de distribuição, continua estável.
- O monitor recomenda **investigar e re-treinar**. Um drift localizado fica diluído na média: monitorar só o global deixaria passar.
- `tempo_relacionamento_meses` muda naturalmente com o tempo (clientes envelhecem) e é marcado como drift esperado, sem disparar alerta.

**Tela publicada (Streamlit Cloud)**: a pasta `app/` tem `requirements.txt` próprio (sem PyTorch) e usa o retrato `demo/scores_atuais.csv`. Com a chave da OpenAI nos *Secrets*, o plano é gerado pela IA, com **limite de 5 planos por sessão** (`LIMITE_PLANOS_IA`) para proteger o crédito; sem a chave, funciona no modo regras.

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python -m churn_b2b.simulacao      # gera os dados em data/            (~10 s)
python -m churn_b2b.treinar        # treina e compara os modelos       (~1 min)
python -m churn_b2b.pontuar        # pontua a carteira atual + motivos (~15 s)

# Etapa 2 — RAG (crie um arquivo .env com OPENAI_API_KEY=sk-... ou use --embeddings local)
python -m churn_b2b.rag.base               # indexa as políticas no ChromaDB
python -m churn_b2b.rag.contexto C00700    # risco + motivos + políticas de um cliente
python -m churn_b2b.rag.avaliar --comparar # acerto@1, acerto@3 e MRR (local x OpenAI)

# Etapa 3 — agente, API e tela
python -m churn_b2b.agente.plano C00700    # plano com LLM (ou --modo regras, sem API)
uvicorn churn_b2b.api:app --reload         # API em http://localhost:8000/docs
streamlit run app/streamlit_app.py         # tela

# Etapa 4 — MLOps
python -m churn_b2b.monitoramento.drift    # PSI + relatório Evidently em reports/
docker build -t churn-b2b-api . && docker run -p 8000:8000 churn-b2b-api
pytest                             # testes

mlflow ui --backend-store-uri sqlite:///mlflow.db   # abre http://localhost:5000
```

## Estrutura

```
churn_b2b/
  config.py       datas de corte, caminhos, MLflow
  simulacao.py    gera clientes, pedidos e atendimentos (com causas de churn e drift proposital)
  features.py     retrato do cliente na data de corte + rótulo (sem vazamento)
  modelos.py      regressão logística, LightGBM, rede neural PyTorch (early stopping), calibração
  treinar.py      busca de hiperparâmetros, MLflow, seleção, relatório
  explicar.py     SHAP → motivos em português + tema de retenção
  pontuar.py      risco, faixa e motivos da carteira atual
  rag/
    documentos.py carrega as políticas e fatia por seção
    embeddings.py OpenAI ou local (sem API)
    base.py       indexação no ChromaDB + busca com filtro por tema e segmento
    contexto.py   pacote do cliente: risco + motivos + políticas (entrada do LLM)
    avaliar.py    acerto@k e MRR do recuperador
  agente/
    plano.py      LangChain + OpenAI (saída estruturada) e modo regras
    validacao.py  guardrails: citações, alçadas, POL-02, POL-10
  api.py          API FastAPI
  monitoramento/
    drift.py      PSI global e por cidade + relatório Evidently
app/              tela Streamlit (+ requirements próprio para o Streamlit Cloud)
.github/workflows/ci.yml   testes + build e teste do Docker + publicação no GHCR
Dockerfile        imagem da API (requirements-api.txt)
demo/             retrato da carteira pontuada, usado pela tela publicada
politicas/        10 políticas fictícias + perguntas de avaliação
tests/            vazamento, rótulo, treino e explicação
reports/          métricas e curva PR
models/           modelo campeão + metadata.json
```

## Sobre a simulação

Cada cliente tem fatores ocultos (sensibilidade a preço, risco financeiro, distância do CD). Quem vai sair recebe uma causa (serviço, preço, financeiro ou "silencioso"), e nos meses anteriores aparecem sinais coerentes com ela. Para o problema não ficar fácil demais, há churns silenciosos (sem aviso) e clientes que esfriam por um tempo e voltam (falsos alarmes). Nos 3 últimos meses há um **choque logístico nas cidades distantes**, de propósito, para o monitoramento de drift detectar na Etapa 4.

---
Autor: João Carlos · Cascavel/PR

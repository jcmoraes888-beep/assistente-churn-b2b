# Roteiro das próximas etapas

## Etapa 2 — RAG (base de retenção) ✅ concluída
- Escrever ~10 documentos curtos de política da distribuidora fictícia: descontos por segmento,
  renegociação de prazo, prioridade de rota para cidades distantes, troca de produto, visita do vendedor,
  limites de alçada (quem aprova o quê).
- Fatiar (chunks), gerar embeddings (OpenAI `text-embedding-3-small`) e salvar no ChromaDB.
- Consulta: `tema_principal` + perfil do cliente → trechos de política mais relevantes.
- Avaliar: para 20 perguntas, o trecho certo está entre os 3 recuperados?

## Etapa 3 — LLM + API ✅ concluída
- LangChain: prompt = risco + motivos SHAP + trechos do RAG → plano de ação + mensagem para o cliente.
- Saída estruturada (Pydantic): ação, responsável, prazo, mensagem, política citada.
- FastAPI: `/clientes/{id}/risco`, `/clientes/{id}/plano`, `/carteira?faixa=Alto`.
- Streamlit: lista de clientes em risco → clique → motivos + plano gerado.

## Etapa 4 — MLOps
- Dockerfile da API; GitHub Actions rodando `pytest` a cada push e publicando a imagem.
- Deploy da API no Render e da tela no Streamlit Cloud.
- Evidently: relatório de drift comparando `referencia_teste.csv` com a carteira atual
  (deve acusar o choque logístico nas cidades distantes).
- Re-treino agendado com registro no MLflow Model Registry.

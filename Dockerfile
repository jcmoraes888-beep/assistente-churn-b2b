# API do Assistente de Churn B2B
# build:  docker build -t churn-b2b-api .
# rodar:  docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... churn-b2b-api
#         (sem a chave, a busca usa embeddings locais e o plano sai no modo regras)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY churn_b2b/ churn_b2b/
COPY politicas/ politicas/
COPY demo/ demo/

# usuário sem privilégios
RUN useradd --create-home app && mkdir -p data && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/saude')"
CMD ["uvicorn", "churn_b2b.api:app", "--host", "0.0.0.0", "--port", "8000"]

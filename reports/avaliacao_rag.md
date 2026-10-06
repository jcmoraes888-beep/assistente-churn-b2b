# Avaliação do RAG

22 perguntas com a política esperada definida à mão (`politicas/perguntas_avaliacao.json`).

| embeddings   | filtro_por_tema   |   acerto@1 |   acerto@3 |   MRR |
|:-------------|:------------------|-----------:|-----------:|------:|
| local        | não               |      0.727 |      0.955 | 0.842 |
| local        | sim               |      0.955 |      1     | 0.977 |
| openai       | não               |      0.909 |      1     | 0.955 |
| openai       | sim               |      1     |      1     | 1     |
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "cidade_match": 1,
      "tem_formacao": 1,
      "pretensao_salarial": 4000,
      "nivel_ingles_score": 3,
      "ingles_intermediario_ou_mais": 1,
      "conhecimento_python": 1,
      "conhecimento_java": 1,
      "conhecimento_sql": 1,
      "conhecimento_excel": 1,
      "conhecimento_c#": 0,
      "conhecimento_docker": 1,
      "conhecimento_git": 1,
      "conhecimento_linux": 1,
      "conhecimento_spring": 1,
      "qtd_cursos": 2,
      "tem_certificacao": 1
    }
  }'
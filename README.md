# Decision Datathon

**Alunos:**
- Lucas Araújo
- Lucas Martins

## Descrição

Este projeto tem como objetivo construir um pipeline de Machine Learning Engineer para previsão de contratação de candidatos em vagas de trabalho, utilizando dados de prospects, candidatos e vagas. O sistema realiza desde a preparação e engenharia de atributos dos dados até o treinamento de um modelo de machine learning (TensorFlow) e disponibilização de uma API para inferência.

## Sobre o Modelo e Funcionamento

O modelo utilizado é uma rede neural densa (feedforward) implementada com TensorFlow/Keras. O pipeline realiza as seguintes etapas:

- **Engenharia de Atributos:** Geração de variáveis relevantes a partir dos dados brutos, como match de cidade, conhecimentos técnicos, nível de inglês, quantidade de cursos, presença de certificações, entre outros.
- **Pré-processamento:** Seleção de features numéricas, normalização dos dados com StandardScaler e balanceamento das classes utilizando SMOTE para lidar com desbalanceamento.
- **Arquitetura:**
  - Camada densa com 128 neurônios (ReLU) + BatchNormalization + Dropout (0.5)
  - Camada densa com 64 neurônios (ReLU) + Dropout (0.3)
  - Camada de saída com 1 neurônio (sigmoid) para classificação binária (contratado ou não)
- **Treinamento:**
  - Função de perda: binary_crossentropy
  - Otimizador: Adam
  - Métricas: accuracy, AUC, precision, recall
  - Ajuste de pesos de classe para compensar desbalanceamento
- **Avaliação:**
  - Relatório de classificação, matriz de confusão e AUC
- **Inferência:**
  - O modelo recebe um dicionário de features e retorna a probabilidade de contratação do candidato.

A API expõe endpoints para predição (`/predict`) e consulta das features esperadas (`/features`).

## Interpretação do Resultado da Predição

De acordo com as habilidades informadas, o sistema retorna uma porcentagem representando a chance de o candidato ser contratado pela Decision. Existe uma margem de oscilação no resultado: por exemplo, se um candidato tem cerca de 80% de chance de ser contratado, ao rodar a predição novamente com os mesmos dados, o valor pode variar levemente (por exemplo, entre 80% e 85%). Essa variação é proposital e serve para simular a incerteza natural de processos seletivos, mas a resposta sempre ficará próxima da faixa real prevista pelo modelo.

## Exemplo de Análise de Resultado

Suponha que você envie o seguinte payload para o endpoint de predição hospedado em:

`https://mledatathondecision-staging.up.railway.app/predict`

```json
{
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
}
```

A resposta da API será algo como:

```json
{
  "probabilidade_contratacao": 83.5
}
```

**Como analisar:**

- O valor retornado indica a chance percentual do candidato ser contratado, de acordo com as informações fornecidas.
- Pequenas variações podem ocorrer a cada requisição, mas o valor sempre estará próximo da faixa real prevista pelo modelo.
- Por exemplo, se o resultado for 83.5, significa que, segundo o modelo, o candidato tem aproximadamente 83% de chance de ser contratado para a vaga analisada.

## Estrutura do Projeto

- `app/`
  - `model/decision_model.py`: Implementação do pipeline de modelagem, treinamento, avaliação e serialização do modelo.
  - `utils/features.py`: Engenharia de atributos, extração e transformação de variáveis relevantes.
  - `utils/preparator.py`: Preparação e unificação dos dados brutos (candidatos, prospects, vagas).
- `data/`: Pasta esperada para os arquivos de entrada (`prospects.zip`, `applicants.zip`, `vagas.zip`).
- `main.py`: API FastAPI para servir o modelo treinado.
- `train_model.py`: Script para executar todo o pipeline de treinamento.
- `Dockerfile` e `run_docker.sh`: Infraestrutura para execução e deploy via Docker.
- `post_example.sh`: Exemplo de requisição para a API.

## Pipeline

1. **Preparação dos Dados**: Unificação e limpeza dos dados de candidatos, prospects e vagas.
2. **Engenharia de Atributos**: Criação de variáveis como match de cidade, conhecimentos técnicos, nível de inglês, quantidade de cursos, presença de certificações, etc.
3. **Treinamento**: Construção do dataset, balanceamento com SMOTE, normalização, treinamento de rede neural densa (TensorFlow).
4. **Avaliação**: Métricas de performance, relatório de classificação, matriz de confusão e AUC.
5. **API**: Disponibilização de endpoint `/predict` para inferência e `/features` para consulta das features esperadas.

## Como Executar

### 1. Treinamento do Modelo

Certifique-se de que os arquivos de dados estejam em `./data`:
- `prospects.zip`
- `applicants.zip`
- `vagas.zip`

Execute o pipeline de treinamento:
```bash
poetry install
poetry run python train_model.py
```

### 2. Subindo a API com Docker

Para buildar e rodar a API:
```bash
bash run_docker.sh
```
A API ficará disponível em `http://localhost:8000`.

### 3. Testando a API

Exemplo de requisição:
```bash
curl -X POST https://mledatathondecision-staging.up.railway.app/predict \
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
```

#### Exemplo de Payload JSON

```json
{
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
}
```

## Dependências Principais

- Python >= 3.10
- polars
- tensorflow
- scikit-learn
- fastapi
- uvicorn
- imbalanced-learn
- poetry

## Observação

A API está disponível publicamente em: [https://mledatathondecision-staging.up.railway.app/predict](https://mledatathondecision-staging.up.railway.app/predict)

O deploy foi realizado utilizando a plataforma Railway.

#!/bin/bash

IMAGE_NAME="decision-datathon"
TAG="latest"
CONTAINER_NAME="decision-datathon-api"

DATA_DIR="./data"
OUTPUT_DIR="./app/model/output"

mkdir -p "$OUTPUT_DIR"

echo "Buildando imagem Docker: $IMAGE_NAME:$TAG"
docker build -t $IMAGE_NAME:$TAG .

if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
  echo "Parando e removendo container anterior..."
  docker rm -f $CONTAINER_NAME
fi

# echo "Subindo container: $CONTAINER_NAME"
# docker run -d \
#   --name $CONTAINER_NAME \
#   -v "$(pwd)/$DATA_DIR:/app/data" \
#   -v "$(pwd)/$OUTPUT_DIR:/app/model/output" \
#   -p 8000:8000 \
#   $IMAGE_NAME:$TAG


docker run -dit \
  --name $CONTAINER_NAME \
  -v "./data:/layer/data" \
  -p 8000:8000 \
  $IMAGE_NAME:$TAG 



# curl -X POST http://127.0.0.1:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{
#     "data": {
#       "cidade_match": 1,
#       "tem_formacao": 1,
#       "pretensao_salarial": 4000,
#       "nivel_ingles_score": 3,
#       "ingles_intermediario_ou_mais": 1,
#       "conhecimento_python": 1,
#       "conhecimento_java": 1,
#       "conhecimento_sql": 1,
#       "conhecimento_excel": 1,
#       "conhecimento_c#": 0,
#       "conhecimento_docker": 1,
#       "conhecimento_git": 1,
#       "conhecimento_linux": 1,
#       "conhecimento_spring": 1,
#       "qtd_cursos": 2,
#       "tem_certificacao": 1
#     }
#   }'
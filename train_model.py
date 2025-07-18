from app.utils.preparator import DataPreparator
from app.utils.features import FeatureEngineer
from app.model.decision_model import DatasetBuilder, TensorflowModel

import logging

def train():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Iniciando processo de treinamento...")

    preparator = DataPreparator(
        prospects_path="./data/prospects.zip",
        applicants_path="./data/applicants.zip",
        vagas_path="./data/vagas.zip"
    )

    logging.info("Carregando e preparando os dados...")
    df = preparator.run()
    logging.info("Dados carregados com sucesso.")

    logging.info("Iniciando engenharia de atributos...")
    engineer = FeatureEngineer(df)
    df_fe = engineer.transform()
    logging.info(f"Feature engineering concluída. Shape final: {df_fe.shape}")

    dataset = DatasetBuilder(df_fe)
    logging.info("Construindo dataset para treino...")

    try:
        X_train, X_test, y_train, y_test = dataset.build()
    except ValueError as e:
        logging.error(f"Erro ao construir dataset: {e}")
        return

    logging.info("Dataset particionado com sucesso.")
    logging.info("Inicializando e treinando modelo TensorFlow...")

    model = TensorflowModel()
    model.build(X_train.shape[1])
    model.train(X_train, y_train, epochs=50, batch_size=32)

    logging.info("Treinamento concluído. Avaliando modelo...")
    model.evaluate(X_test, y_test)
    model.save()

    logging.info("Pipeline de treinamento executado com sucesso.")

if __name__ == "__main__":
    train()
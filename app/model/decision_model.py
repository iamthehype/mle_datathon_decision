
import polars as pl
import numpy as np
import tensorflow as tf
import joblib
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetBuilder:
    def __init__(self, df: pl.DataFrame, target_col: str = "foi_contratado"):
        self.df = df
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.feature_columns = []

    def _select_features(self):
        exclude = [self.target_col, "codigo_profissional", "nome", "vaga_id"]
        features = [
            col for col in self.df.columns
            if self.df[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]
            and col not in exclude
        ]
        logging.info(f"Total de features numéricas: {len(features)}")
        self.feature_columns = features
        return features

    def _filter_low_variance(self, X: np.ndarray, threshold=1e-5):
        variances = np.var(X, axis=0)
        mask = variances > threshold
        filtered_features = [f for f, keep in zip(self.feature_columns, mask) if keep]
        return X[:, mask], filtered_features

    def build(self):
        if self.target_col not in self.df.columns:
            raise ValueError(f"Coluna target '{self.target_col}' não encontrada no DataFrame.")

        logging.info("Pré-visualização do DataFrame:\n" + str(self.df.head(5)))

        features = self._select_features()
        X = self.df.select(features).to_numpy()
        y = self.df.select(self.target_col).to_numpy().ravel()

        logging.info(f"Distribuição original de classes: {np.bincount(y.astype(int))}")

        # Injeção de alguns registros positivos se houver poucos
        if np.sum(y == 1) < 5:
            logging.warning("Poucos exemplos positivos. Serão adicionados exemplos sintéticos.")
            positive_indices = np.where(y == 1)[0]
            if len(positive_indices) > 0:
                num_duplicates = 10 - len(positive_indices)
                indices_to_duplicate = np.random.choice(positive_indices, size=num_duplicates, replace=True)
                X = np.vstack([X, X[indices_to_duplicate]])
                y = np.concatenate([y, y[indices_to_duplicate]])
                logging.info(f"Exemplos positivos após adição: {np.sum(y == 1)}")

        if len(np.unique(y)) <= 1:
            raise ValueError("A variável target possui apenas uma classe. Treinamento não é possível.")

        X, self.feature_columns = self._filter_low_variance(X)
        X_scaled = self.scaler.fit_transform(X)

        joblib.dump(self.scaler, OUTPUT_DIR / "scaler.pkl")
        joblib.dump(self.feature_columns, OUTPUT_DIR / "features.pkl")
        logging.info(f"Scaler e {len(self.feature_columns)} features salvos em {OUTPUT_DIR}")

        config = {
            "features": self.feature_columns,
            "scaler": "StandardScaler",
            "model_architecture": "Dense-[128,64]",
            "dropout": [0.5, 0.3],
            "epochs": 50,
            "batch_size": 32
        }
        with open(OUTPUT_DIR / "train_config.json", "w") as f:
            json.dump(config, f, indent=2)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        if len(np.unique(y_train)) > 1:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"Distribuição após SMOTE: {np.bincount(y_train)}")
        else:
            logging.warning("SMOTE não aplicado: y_train contém apenas uma classe.")

        return X_train, X_test, y_train, y_test

class TensorflowModel:
    def __init__(self):
        self.model = None

    def build(self, input_shape):
        logging.info(f"Construindo modelo com input shape: {input_shape}")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name="auc"),
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall")]
        )
        self.model = model
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(weights))
        logging.info(f"Pesos de classe usados no treino: {class_weights}")

        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            class_weight=class_weights
        )

    def evaluate(self, X_test, y_test, threshold=0.3):
        logging.info("Avaliando modelo com threshold ajustado...")
        results = self.model.evaluate(X_test, y_test, verbose=1)
        y_probs = self.model.predict(X_test)
        y_pred = (y_probs > threshold).astype(int)

        logging.info("\n" + classification_report(y_test, y_pred))
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        logging.info(f"AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
        return results

    def save(self, path=OUTPUT_DIR / "modelo_final.h5"):
        if self.model:
            self.model.save(str(path))
            logging.info(f"Modelo salvo em {path}")

    def load(self, path=OUTPUT_DIR / "modelo_final.h5"):
        self.model = tf.keras.models.load_model(str(path))
        return self.model

    def predict(self, df_novo: pl.DataFrame):
        scaler = joblib.load(OUTPUT_DIR / "scaler.pkl")
        features = joblib.load(OUTPUT_DIR / "features.pkl")

        missing_cols = [col for col in features if col not in df_novo.columns]
        for col in missing_cols:
            df_novo = df_novo.with_columns(pl.lit(0).alias(col))

        df_novo = df_novo.select(features)
        X_new = df_novo.to_numpy()
        X_scaled = scaler.transform(X_new)

        return self.model.predict(X_scaled)

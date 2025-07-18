import polars as pl
import json
from pathlib import Path
from zipfile import ZipFile

class DataPreparator:
    def __init__(self, applicants_path: str, prospects_path: str, vagas_path: str):
        self.applicants_path = Path(applicants_path)
        self.prospects_path = Path(prospects_path)
        self.vagas_path = Path(vagas_path)

    def _read_json_from_path(self, path: Path) -> dict:
        if path.suffix == ".zip":
            with ZipFile(path, "r") as zipf:
                json_filename = zipf.namelist()[0]
                with zipf.open(json_filename) as f:
                    return json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _load_flat_json_df(self, path: Path, id_field: str) -> pl.DataFrame:
        data = self._read_json_from_path(path)

        records = []
        for id_, content in data.items():
            flat_record = {id_field: id_}
            for section, values in content.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        flat_record[f"{section}_{k}"] = v
                else:
                    flat_record[section] = values
            records.append(flat_record)

        return pl.DataFrame(records)

    def _load_prospects(self) -> pl.DataFrame:
        data = self._read_json_from_path(self.prospects_path)

        prospect_records = []
        for vaga_id, vaga_info in data.items():
            for prospect in vaga_info.get("prospects", []):
                prospect["vaga_id"] = vaga_id
                prospect_records.append(prospect)

        df = pl.DataFrame(prospect_records).with_columns([
            pl.col("situacao_candidado").cast(str).str.to_lowercase().str.strip_chars().alias("situacao_candidado"),
            pl.when(pl.col("situacao_candidado").str.contains("contratado"))
              .then(1)
              .otherwise(0)
              .alias("foi_contratado")
        ])

        return df.rename({"codigo": "codigo_profissional"})

    def run(self) -> pl.DataFrame:
        df_applicants = self._load_flat_json_df(self.applicants_path, "codigo_profissional")
        df_vagas = self._load_flat_json_df(self.vagas_path, "codigo_vaga")
        df_prospects = self._load_prospects()

        df_merged = df_prospects.join(df_applicants, on="codigo_profissional", how="left")
        df_merged = df_merged.join(df_vagas, left_on="vaga_id", right_on="codigo_vaga", how="left")

        return df_merged
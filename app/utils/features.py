import polars as pl

class FeatureEngineer:
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def _safe_extract_number(self, col: pl.Series) -> pl.Series:
        return (
            col.str.replace_all(r"[^0-9,\\.]", "")
               .str.replace_all(",", ".")
               .cast(pl.Float64, strict=False)
        )

    def _extract_conhecimentos(self, df: pl.DataFrame) -> pl.DataFrame:
        tecnologias = ["python", "java", "sql", "excel", "docker", "git", "linux", "spring"]

        for tech in tecnologias:
            df = df.with_columns([
                df["informacoes_profissionais_conhecimentos_tecnicos"]
                .fill_null("")
                .str.to_lowercase()
                .str.contains(fr"\b{tech}\b")
                .cast(pl.Int8)
                .alias(f"conhecimento_{tech}")
            ])
        return df

    def _count_cursos(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col("formacao_e_idiomas_cursos")
            .fill_null("")
            .str.strip_chars()
            .str.split(",")
            .list.len()
            .alias("qtd_cursos")
        ])

    def _presenca_certificacao(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col("informacoes_profissionais_certificacoes")
            .fill_null("")
            .str.strip_chars()
            .str.split(",")
            .list.len()
            .gt(0)
            .cast(pl.Int8)
            .alias("tem_certificacao")
        ])

    def transform(self) -> pl.DataFrame:
        df = self.df

        # Match cidade do candidato com a cidade da vaga
        df = df.with_columns([
            (pl.col("informacoes_pessoais_endereco").fill_null("").str.to_lowercase().str.strip_chars()
             == pl.col("perfil_vaga_cidade").fill_null("").str.to_lowercase().str.strip_chars())
            .cast(pl.Int8)
            .alias("cidade_match")
        ])

        # Formação presente
        df = df.with_columns([
            pl.col("formacao_e_idiomas_nivel_academico").is_not_null().cast(pl.Int8).alias("tem_formacao")
        ])

        # Pretensão salarial
        df = df.with_columns([
            self._safe_extract_number(pl.col("informacoes_profissionais_remuneracao")).alias("pretensao_salarial")
        ])

        # Nível de inglês e derivada binária
        idioma_ingles = df.select([
            pl.col("formacao_e_idiomas_nivel_ingles").fill_null("nenhum").str.to_lowercase()
        ]).to_series(0).map_elements(lambda x: {
            "nenhum": 0, "básico": 1, "intermediário": 2, "avançado": 3, "fluente": 4
        }.get(x.strip(), 0))

        df = df.with_columns([
            pl.Series("nivel_ingles_score", idioma_ingles),
            pl.Series("nivel_ingles_score", idioma_ingles).ge(2).cast(pl.Int8).alias("ingles_intermediario_ou_mais")
        ])

        # Conhecimentos técnicos
        df = self._extract_conhecimentos(df)

        # Cursos
        df = self._count_cursos(df)

        # Certificações
        df = self._presenca_certificacao(df)

        return df
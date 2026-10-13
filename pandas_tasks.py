"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic."""
    df = pd.read_csv(data.csv_path)

    row_count = len(df)

    missing_by_column = {
        str(col): int(count) for col, count in df.isna().sum().to_dict().items()
    }

    adults_over_30_count = int((df['Age'].dropna() > 30).sum())

    mean_age_by_pclass = {
        int(k): float(v)
        for k, v in df.groupby('Pclass')['Age'].mean().to_dict().items()
    }

    survival_rate_by_pclass = {
        int(k): float(v)
        for k, v in df.groupby('Pclass')['Survived'].mean().to_dict().items()
    }

    highest_fares = df['Fare'].nlargest(5).tolist()

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )

"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt."""
    df = pd.read_csv(data.csv_path, sep='\t', na_values=['NA'])

    men_df = df[df['Gender'] == 'Male']
    women_df = df[df['Gender'] == 'Female']

    men_count = len(men_df)
    women_count = len(women_df)

    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    target = 'MRI_Count'

    men_mri_correlation = {}
    for feat in features:
        corr = men_df[feat].corr(men_df[target], method='pearson')
        men_mri_correlation[feat] = float(corr) if not pd.isna(corr) else 0.0

    women_mri_correlation = {}
    for feat in features:
        corr = women_df[feat].corr(women_df[target], method='pearson')
        women_mri_correlation[feat] = float(corr) if not pd.isna(corr) else 0.0

    all_correlations = {}
    for feat in features:
        all_correlations[feat] = max(
            abs(men_mri_correlation[feat]),
            abs(women_mri_correlation[feat]),
        )

    strongest_mri_feature = max(all_correlations, key=all_correlations.get)

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )

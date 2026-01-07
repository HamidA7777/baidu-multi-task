import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

click_model_data_path = 'click-model-bar-chart-data.csv'
multi_task_model_with_position_debiasing_data_path = 'mulit-task-with-debiasing-one-logit-bar-chart-data.csv'
multi_task_model_without_debiasing_data_path = 'multi-task-without-debiasing-bar-chart-data.csv'
multi_task_model_per_task_position_debiasing_data_path = 'multi-task-with-per-task-debiasing-bar-chart-data.csv'

click_model_stds_path = 'click-model-stds.csv'
multi_task_model_with_position_debiasing_stds = 'multi-task-with-debiasing-one-logit-stds.csv'
multi_task_model_without_debiasing_stds = 'multi-task-without-debiasing-stds.csv'
multi_task_model_per_task_position_debiasing_stds = 'multi-task-with-per-task-debiasing-stds.csv'

base_path = os.path.dirname(os.path.abspath(__file__))

column_mapping = {
    'ranking_metrics.MRR@10': 'MRR@10',
    'ranking_metrics.DCG@5': 'DCG@5',
    'ranking_metrics.DCG@3': 'DCG@3',
    'ranking_metrics.DCG@1': 'DCG@1',
    'ranking_metrics.DCG@10': 'DCG@10',
    'ranking_metrics.nDCG@10': 'NDCG@10',
}

df_click_model = pd.read_csv(os.path.join(base_path, '..', 'utils', click_model_data_path))
df_multi_task_model_with_position_debiasing = pd.read_csv(os.path.join(base_path, '..', 'utils', multi_task_model_with_position_debiasing_data_path))
df_multi_task_model_without_debiasing = pd.read_csv(os.path.join(base_path, '..', 'utils', multi_task_model_without_debiasing_data_path))
df_multi_task_model_per_task_position_debiasing = pd.read_csv(os.path.join(base_path, '..', 'utils', multi_task_model_per_task_position_debiasing_data_path))

df_click_model.rename(columns=lambda c: column_mapping[c] if c in column_mapping else c, inplace=True)
df_multi_task_model_with_position_debiasing.rename(columns=lambda c: column_mapping[c] if c in column_mapping else c, inplace=True)
df_multi_task_model_without_debiasing.rename(columns=lambda c: column_mapping[c] if c in column_mapping else c, inplace=True)
df_multi_task_model_per_task_position_debiasing.rename(columns=lambda c: column_mapping[c] if c in column_mapping else c, inplace=True)

df_click_model_stds = pd.read_csv(os.path.join(base_path, '..', 'utils', click_model_stds_path))
df_multi_task_model_with_position_debiasing_stds = pd.read_csv(os.path.join(base_path, '..', 'utils', multi_task_model_with_position_debiasing_stds))
df_multi_task_model_without_debiasing_stds = pd.read_csv(os.path.join(base_path, '..', 'utils', multi_task_model_without_debiasing_stds))
df_multi_task_model_per_task_position_debiasing_stds = pd.read_csv(os.path.join(base_path, '..', 'utils', multi_task_model_per_task_position_debiasing_stds))


def transform_stds_df(df: pd.DataFrame):
    columns = ['Metric', 'Std']
    data = []
    df['Metric'] = df['Metric'].apply(lambda metric: metric.upper())
    ranking_metrics = df['Metric'].unique()
    for metric in ranking_metrics:
        data.append({'Metric': metric, 'Std': df['Std'][df['Metric'] == metric].values[0]})
    return pd.DataFrame(data=data)


def transform_dataframe(df: pd.DataFrame):
    columns = ['Metric', 'Score']
    ranking_metrics = [
        'MRR@10',
        'DCG@5',
        'DCG@3',
        'DCG@1',
        'DCG@10',
        'NDCG@10',
    ]
    data = []
    for metric in ranking_metrics:
        data.append({'Metric': metric, 'Score': df[metric].values[0]})
    return pd.DataFrame(data=data)


df_click_model = transform_dataframe(df_click_model)
df_multi_task_model_with_position_debiasing = transform_dataframe(df_multi_task_model_with_position_debiasing)
df_multi_task_model_without_debiasing = transform_dataframe(df_multi_task_model_without_debiasing)
df_multi_task_model_per_task_position_debiasing = transform_dataframe(df_multi_task_model_per_task_position_debiasing)

df_click_model.columns = ['Metric', 'Click-only model']
df_multi_task_model_with_position_debiasing.columns = ['Metric', 'Multi-task model with position debiasing']
df_multi_task_model_without_debiasing.columns = ['Metric', 'Multi-task model without debiasing']
df_multi_task_model_per_task_position_debiasing.columns = ['Metric', 'Multi-task model with per-task position debiasing']

merged = df_click_model.merge(df_multi_task_model_with_position_debiasing, on='Metric').merge(df_multi_task_model_without_debiasing, on='Metric') \
            .merge(df_multi_task_model_per_task_position_debiasing, on='Metric')
merged.set_index('Metric', inplace=True)

df_click_model_stds = transform_stds_df(df_click_model_stds)
df_multi_task_model_with_position_debiasing_stds = transform_stds_df(df_multi_task_model_with_position_debiasing_stds)
df_multi_task_model_without_debiasing_stds = transform_stds_df(df_multi_task_model_without_debiasing_stds)
df_multi_task_model_per_task_position_debiasing_stds = transform_stds_df(df_multi_task_model_per_task_position_debiasing_stds)

df_click_model_stds.columns = ['Metric', 'Click-only stds']
df_multi_task_model_with_position_debiasing_stds.columns = ['Metric', 'Multi-task model with position debiasing stds']
df_multi_task_model_without_debiasing_stds.columns = ['Metric', 'Multi-task model without debiasing stds']
df_multi_task_model_per_task_position_debiasing_stds.columns = ['Metric', 'Multi-task model with per-task position debiasing stds']

merged_stds = df_click_model_stds.merge(df_multi_task_model_with_position_debiasing_stds, on='Metric').\
    merge(df_multi_task_model_without_debiasing_stds, on='Metric').merge(df_multi_task_model_per_task_position_debiasing_stds, on='Metric')

merged_stds.set_index('Metric', inplace=True)

metrics = merged.index.tolist()
bar_width = 0.15
x = np.arange(len(metrics))

fig, ax = plt.subplots(figsize=(12, 8))

ax.barh(x - 2*bar_width, merged['Click-only model'], xerr=merged_stds['Click-only stds'], height=bar_width, label='Click-only model')
ax.barh(x - bar_width, merged['Multi-task model with position debiasing'], xerr=merged_stds['Multi-task model with position debiasing stds'], 
    height=bar_width, label='Multi-task model with position debiasing')
ax.barh(x, merged['Multi-task model with per-task position debiasing'], xerr=merged_stds['Multi-task model with per-task position debiasing stds'],
    height=bar_width, label='Multi-task model with per-task position debiasing')
ax.barh(x + 2*bar_width, merged['Multi-task model without debiasing'], xerr=merged_stds['Multi-task model without debiasing stds'],
    height=bar_width, label='Multi-task model without debiasing')

ax.set_yticks(x)
ax.set_yticklabels(metrics)
ax.invert_yaxis()
ax.set_xlabel('Score')
ax.set_title('Ranking Metrics Comparison Across Models')
ax.legend()

plt.tight_layout()
plt.show()

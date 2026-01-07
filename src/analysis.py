import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
import json

from ranking_metrics import REL_METRICS_CUSTOM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, '..', 'utils', 'multi-task_evaluation_scores.csv')

df = pd.read_csv(file_path)

def evaluate_ranking_metrics_raw_bert(df: pd.DataFrame, ranking_feature):
    results = {metric: [] for metric in REL_METRICS_CUSTOM}

    for query_id, group in df.groupby(by='query_id'):
        sorted_group = group.sort_values(by=ranking_feature, ascending=False)

        labels = sorted_group['label'].values
        scores = sorted_group[ranking_feature].values

        for name, metric in REL_METRICS_CUSTOM.items():
            metric_value = float(metric(scores, labels))
            results[name].append(metric_value)

    average_results = {name: sum(ranking_scores) / len(ranking_scores) for name, ranking_scores in results.items()}

    return average_results

df['click_scores'] = df['click_scores'].apply(lambda x : F.sigmoid(torch.tensor(x)).item())
df['skip_scores'] = df['skip_scores'].apply(lambda x : F.sigmoid(torch.tensor(x)).item())

def plot_labels_to_signal(df: pd.DataFrame, feature: str):
    name_mapper = {
        "click_scores": "Click",
        "skip_scores": "Skip",
        "dwell_time_scores": "Dwell-time"
    }

    def mean_ci(x):
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        n = len(x)
        ci = 1.96 * std / np.sqrt(n)
        return mean, ci

    grouped = df.groupby("label")[feature].apply(mean_ci)

    labels = grouped.index.values
    means = [v[0] for v in grouped]
    cis = [v[1] for v in grouped]

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        labels,
        means,
        yerr=cis,
        fmt='o-',
        capsize=5
    )

    plt.xticks(labels)
    plt.xlabel("Annotator Relevance Label")
    if feature in ['skip_scores', 'click_scores']:
        plt.ylabel(f"Predicted {name_mapper[feature]} Probability")
        plt.title(f"{name_mapper[feature]} Probability vs. Relevance Label")
    else:
        plt.ylabel(f"Predicted {name_mapper[feature]}")
        plt.title(f"{name_mapper[feature]} vs. Relevance Label")

    plt.grid(True)
    plt.tight_layout()
    plt.show()


ultr_feedback_path = os.path.join(BASE_DIR, '..', 'utils', 'ultr_feedback_stats.json')

with open(ultr_feedback_path, "r") as f:
    stats = json.load(f)

docs_processed = stats["docs_processed"]
clicks = stats["clicks"][1:]
impressions = stats["impressions"][1:]
skips = stats["skips"][1:]
examined = stats["examined"][1:]
dwell_sum = stats["dwell_sum"][1:]
dwell_count = stats["dwell_count"][1:]

clicks = np.array(clicks)
impressions = np.array(impressions)
skips = np.array(skips)
examined = np.array(examined)
dwell_sum = np.array(dwell_sum)
dwell_count = np.array(dwell_count)

ctr = clicks / np.maximum(impressions, 1)
skip_rate = skips / np.maximum(examined, 1)
mean_dwell = dwell_sum / np.maximum(dwell_count, 1)

positions = np.arange(1, len(ctr) + 1)

plt.figure()
plt.plot(positions, ctr, marker="o", label='Click-through rate')
plt.plot(positions, skip_rate, marker="o", label='Skip rate')
plt.plot(positions, mean_dwell, marker="o", label='Average dwell-time (log-scale)')
plt.xlabel("Position")
plt.grid(True)
plt.legend()
plt.show()

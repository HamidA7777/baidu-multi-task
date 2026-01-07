import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dwell_times():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dwell_times_path = os.path.join(BASE_DIR, '..', 'utils', 'dwell_times.csv')

    dwell_times_df = pd.read_csv(dwell_times_path)
    dwell_times_original = dwell_times_df['original']
    dwell_times_transformed = dwell_times_df['transformed']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(dwell_times_original, bins=30, color='skyblue')
    plt.title('Original Dwell-times')
    plt.xlabel('Dwell Time')
    plt.ylabel('Frequency (log scale)')

    plt.yscale('log')

    plt.subplot(1, 2, 2)
    sns.histplot(dwell_times_transformed, bins=30, color='salmon')
    plt.title('Transformed Dwell-times (log1p)')
    plt.xlabel('Dwell Time')
    plt.ylabel('Frequency (log scale)')

    plt.yscale('log')

    plt.tight_layout()
    plt.show()


def transform_dataframe(df: pd.DataFrame, kind_loss):
    df_columns = df.columns
    filtered_columns = [column for column in df_columns if not '_step' in column and not 'MIN' in column and not 'MAX' in column]
    df_filtered = df[filtered_columns]
    df_filtered.columns = ['step', kind_loss]
    return df_filtered


def smooth_series(series, window=100):
    return series.rolling(window=window, min_periods=1).mean()


def plot_losses(kind="Total Loss"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(BASE_DIR, '..', 'utils', 'loss')
    multi_task_path = os.path.join(base_path, "multi-task")
    click_only_path = os.path.join(base_path, "click-model")

    if kind == "Total Loss":
        multi_task_without_debiasing_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-without-debiasing-total_loss.csv'))
        multi_task_with_debiasing_sep_logits_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-separate-logits-total_loss.csv'))
        multi_task_with_debiasing_one_logit_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-one-logit-total_loss.csv'))
        click_only_df = pd.read_csv(os.path.join(click_only_path, 'click-model_total-loss.csv'))

        multi_task_without_debiasing_df = transform_dataframe(multi_task_without_debiasing_df, kind_loss=kind)
        multi_task_with_debiasing_sep_logits_df = transform_dataframe(multi_task_with_debiasing_sep_logits_df, kind_loss=kind)
        multi_task_with_debiasing_one_logit_df = transform_dataframe(multi_task_with_debiasing_one_logit_df, kind_loss=kind)
        click_only_df = transform_dataframe(click_only_df, kind_loss=kind)

        multi_task_without_debiasing_df['smoothed'] = smooth_series(multi_task_without_debiasing_df[kind])
        multi_task_with_debiasing_sep_logits_df['smoothed'] = smooth_series(multi_task_with_debiasing_sep_logits_df[kind])
        multi_task_with_debiasing_one_logit_df['smoothed'] = smooth_series(multi_task_with_debiasing_one_logit_df[kind])
        click_only_df['smoothed'] = smooth_series(click_only_df[kind])

    elif kind == "MLM Loss":
        multi_task_without_debiasing_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-without-debiasing-mlm_loss.csv'))
        multi_task_with_debiasing_sep_logits_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-separate-logits-mlm_loss.csv'))
        multi_task_with_debiasing_one_logit_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-one-logit-mlm_loss.csv'))
        click_only_df = pd.read_csv(os.path.join(click_only_path, 'click-model_mlm-loss.csv'))

        multi_task_without_debiasing_df = transform_dataframe(multi_task_without_debiasing_df, kind_loss=kind)
        multi_task_with_debiasing_sep_logits_df = transform_dataframe(multi_task_with_debiasing_sep_logits_df, kind_loss=kind)
        multi_task_with_debiasing_one_logit_df = transform_dataframe(multi_task_with_debiasing_one_logit_df, kind_loss=kind)
        click_only_df = transform_dataframe(click_only_df, kind_loss=kind)

        multi_task_without_debiasing_df['smoothed'] = smooth_series(multi_task_without_debiasing_df[kind])
        multi_task_with_debiasing_sep_logits_df['smoothed'] = smooth_series(multi_task_with_debiasing_sep_logits_df[kind])
        multi_task_with_debiasing_one_logit_df['smoothed'] = smooth_series(multi_task_with_debiasing_one_logit_df[kind])
        click_only_df['smoothed'] = smooth_series(click_only_df[kind])

    elif kind == "Click Loss":
        multi_task_without_debiasing_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-without-debiasing-click_loss.csv'))
        multi_task_with_debiasing_sep_logits_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-separate-logits-click_loss.csv'))
        multi_task_with_debiasing_one_logit_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-one-logit-click_loss.csv'))
        click_only_df = pd.read_csv(os.path.join(click_only_path, 'click-model_click-loss.csv'))

        multi_task_without_debiasing_df = transform_dataframe(multi_task_without_debiasing_df, kind_loss=kind)
        multi_task_with_debiasing_sep_logits_df = transform_dataframe(multi_task_with_debiasing_sep_logits_df, kind_loss=kind)
        multi_task_with_debiasing_one_logit_df = transform_dataframe(multi_task_with_debiasing_one_logit_df, kind_loss=kind)
        click_only_df = transform_dataframe(click_only_df, kind_loss=kind)

        multi_task_without_debiasing_df['smoothed'] = smooth_series(multi_task_without_debiasing_df[kind])
        multi_task_with_debiasing_sep_logits_df['smoothed'] = smooth_series(multi_task_with_debiasing_sep_logits_df[kind])
        multi_task_with_debiasing_one_logit_df['smoothed'] = smooth_series(multi_task_with_debiasing_one_logit_df[kind])
        click_only_df['smoothed'] = smooth_series(click_only_df[kind])

    elif kind == "Skip Loss":
        multi_task_without_debiasing_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-without-debiasing-skip_loss.csv'))
        multi_task_with_debiasing_sep_logits_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-separate-logits-skip_loss.csv'))
        multi_task_with_debiasing_one_logit_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-one-logit-skip_loss.csv'))

        multi_task_without_debiasing_df = transform_dataframe(multi_task_without_debiasing_df, kind_loss=kind)
        multi_task_with_debiasing_sep_logits_df = transform_dataframe(multi_task_with_debiasing_sep_logits_df, kind_loss=kind)
        multi_task_with_debiasing_one_logit_df = transform_dataframe(multi_task_with_debiasing_one_logit_df, kind_loss=kind)

        multi_task_without_debiasing_df['smoothed'] = smooth_series(multi_task_without_debiasing_df[kind])
        multi_task_with_debiasing_sep_logits_df['smoothed'] = smooth_series(multi_task_with_debiasing_sep_logits_df[kind])
        multi_task_with_debiasing_one_logit_df['smoothed'] = smooth_series(multi_task_with_debiasing_one_logit_df[kind])

    elif kind == "Dwell-time Loss":
        multi_task_without_debiasing_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-without-debiasing-dwell-time_loss.csv'))
        multi_task_with_debiasing_sep_logits_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-separate-logits-dwell-time_loss.csv'))
        multi_task_with_debiasing_one_logit_df = pd.read_csv(os.path.join(multi_task_path, 'multi-task-with-debiasing-one-logit-dwell-time_loss.csv'))

        multi_task_without_debiasing_df = transform_dataframe(multi_task_without_debiasing_df, kind_loss=kind)
        multi_task_with_debiasing_sep_logits_df = transform_dataframe(multi_task_with_debiasing_sep_logits_df, kind_loss=kind)
        multi_task_with_debiasing_one_logit_df = transform_dataframe(multi_task_with_debiasing_one_logit_df, kind_loss=kind)

        multi_task_without_debiasing_df['smoothed'] = smooth_series(multi_task_without_debiasing_df[kind])
        multi_task_with_debiasing_sep_logits_df['smoothed'] = smooth_series(multi_task_with_debiasing_sep_logits_df[kind])
        multi_task_with_debiasing_one_logit_df['smoothed'] = smooth_series(multi_task_with_debiasing_one_logit_df[kind])

    plt.figure(figsize=(12, 6))

    plt.plot(multi_task_without_debiasing_df['step'], multi_task_without_debiasing_df['smoothed'], label='multi-task without debiasing', color='blue')
    plt.plot(multi_task_with_debiasing_sep_logits_df['step'], multi_task_with_debiasing_sep_logits_df['smoothed'], label='multi-task with per-task position debiasing', color='red')
    plt.plot(multi_task_with_debiasing_one_logit_df['step'], multi_task_with_debiasing_one_logit_df['smoothed'], label='multi-task with position debiasing', color='green')

    if kind in ["Total Loss", "MLM Loss", "Click Loss"]:
        plt.plot(click_only_df['step'], click_only_df['smoothed'], label='click-only model', color='black')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f"{kind} Over Time Across Different Models")
    plt.xlabel("Training Step")
    plt.ylabel(f"{kind}")
    plt.legend()
    plt.tight_layout()
    sns.set(style='whitegrid')
    plt.show()


plot_losses(kind="Total Loss")

import os
import csv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from ranking_metrics import REL_METRICS_CUSTOM
from sklearn.model_selection import StratifiedKFold
from config import Config


USE_KFOLD = True
KFolds = 3 if USE_KFOLD else 0

IS_MULTI_TASK = True

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
LEARNING_RATE = 1e-3
EPOCHS = 20


def set_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_evaluation_scores(path=f'{Config.EVALUATION_PATH.value}/{Config.MULTI_TASK_EVALUATION.value}'):
    evaluation_df = pd.read_csv(path)
    return evaluation_df


def create_train_test_split(evaluation_df, test_size=0.8, random_state=1):
    unique_queries = evaluation_df['query_id'].unique()
    train_query_ids, test_query_ids = train_test_split(unique_queries, test_size=test_size, random_state=random_state)
    train_df = evaluation_df[evaluation_df['query_id'].isin(train_query_ids)]
    test_df = evaluation_df[evaluation_df['query_id'].isin(test_query_ids)]
    return train_df, test_df


class MultiTaskFusionDataset(Dataset):
    def __init__(self, df, features, label):
        self.df = df
        self.features = features
        self.label = label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        query_id = self.df.iloc[idx]['query_id']
        bucket = self.df.iloc[idx]['bucket']
        features = torch.tensor(self.df.iloc[idx][self.features].values, dtype=torch.float)
        label = torch.tensor(self.df.iloc[idx][self.label], dtype=torch.float)
        return {
            'query_id': query_id,
            'label': label,
            'bucket': bucket,
            'features': features
        }


class MultiTaskFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_features = 3

        self.multi_task_fusion_model = nn.Sequential(
            nn.Linear(self.n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.multi_task_fusion_model(x)


def train_multi_task_fuse(model, dataloader, optimizer, criterion, epochs,
                        save_path=f'{Config.CHECKPOINT_PATH.value}/best_multi-task-fusion_model.pt'):
    best_loss = float('inf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epochs_without_improvements = 0

    steps = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            input = batch['features'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output.squeeze(-1), label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            steps += 1

        avg_loss = total_loss / len(dataloader)

        if avg_loss < best_loss:
            epochs_without_improvements = 0
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        else:
            epochs_without_improvements += 1

            if epochs_without_improvements >= 5:
                break


def train_multi_task_fuse_kfold(model, dataset, df, kfolds, optimizer, criterion, epochs,
                                save_path=f'{Config.CHECKPOINT_PATH.value}/best_multi-task-fusion_model.pt'):
    best_loss = float('inf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    query_df = df.groupby('query_id').agg({
        'label': 'mean',
    }).reset_index()

    query_counts = df['query_id'].value_counts()
    query_df['query_freq'] = query_df['query_id'].map(query_counts)
    query_df['stratify_bin'] = pd.qcut(query_df['query_freq'], q=kfolds, labels=False)

    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=1)
    query_ids = query_df['query_id'].values
    stratify_bins = query_df['stratify_bin'].values

    results_per_fold = {metric: [] for metric in REL_METRICS_CUSTOM}

    for fold, (train_idx, test_idx) in enumerate(skf.split(query_ids, stratify_bins)):

        epochs_without_improvements = 0
        
        model = MultiTaskFusionModel().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        train_steps = 0
        test_steps = 0

        train_query_ids = query_ids[train_idx]
        test_query_ids = query_ids[test_idx]

        df = df.reset_index(drop=True)
        train_row_idx = df[df['query_id'].isin(train_query_ids)].index.tolist()
        test_row_idx = df[df['query_id'].isin(test_query_ids)].index.tolist()

        train_subset = Subset(dataset, train_row_idx)
        test_subset = Subset(dataset, test_row_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=1)

        fold_best_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                input = batch['features'].to(device)
                label = batch['label'].to(device)

                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output.squeeze(-1), label)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                train_steps += 1

            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    input = batch['features'].to(device)
                    label = batch['label'].to(device)

                    output = model(input)

                    test_loss = criterion(output.squeeze(-1), label).item()

                    total_test_loss += test_loss

                    test_steps += 1

            avg_test_loss = total_test_loss / len(test_loader)

            if avg_test_loss < fold_best_loss:
                fold_best_loss = avg_test_loss
                epochs_without_improvements = 0
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    torch.save(model.state_dict(), save_path)

            else:
                epochs_without_improvements += 1

                if epochs_without_improvements >= 5:
                    break

        model.load_state_dict(torch.load(save_path))
        model.eval()

        evaluate_ranking_metrics_multi_task_per_fold(
            model=model,
            dataloader=test_loader,
            results_dict=results_per_fold
        )

        print(results_per_fold)

    results_std = {metric: np.std(scores) for metric, scores in results_per_fold.items()}

    print(results_std)


def evaluate_ranking_metrics_multi_task_per_fold(model, dataloader, results_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    df = []
    with torch.no_grad():
        for batch in dataloader:
            query_id = batch['query_id']
            label = batch['label']
            bucket = batch['bucket']
            input = batch['features'].to(device)
            fused_scores = model(input)

            for i in range(len(fused_scores)):
                df.append({
                    'query_id': query_id[i].item(),
                    'label': label[i].item(),
                    'bucket': bucket[i].item(),
                    'fused_scores': fused_scores[i].item(),
                })

    df = pd.DataFrame(df)

    fold_results = {metric: [] for metric in REL_METRICS_CUSTOM}
    for query_id, group in df.groupby('query_id'):
        sorted_group = group.sort_values(by='fused_scores', ascending=False)

        labels = sorted_group['label'].values
        scores = sorted_group['fused_scores'].values

        for name, metric_function in REL_METRICS_CUSTOM.items():
            metric_value = float(metric_function(scores, labels))
            fold_results[name].append(metric_value)

    for name, scores in fold_results.items():
        avg = np.mean(scores)
        results_dict[name].append(avg)


def evaluate_multi_task_fuse(model, dataloader, model_path=f'{Config.CHECKPOINT_PATH.value}/best_multi-task-fusion_model.pt', 
    save_path=f'{Config.EVALUATION_PATH.value}/{Config.MULTI_TASK_FUSE_EVALUATION.value}'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', newline='') as fused_evaluation_scores:
        fieldnames = ['query_id', 'label', 'bucket', 'fused_scores']
        writer = csv.DictWriter(fused_evaluation_scores, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for batch in dataloader:
                query_id = batch['query_id']
                label = batch['label']
                bucket = batch['bucket']

                input = batch['features'].to(device)

                fused_scores = model(input)

                for i in range(len(fused_scores)):
                    writer.writerow({
                        'query_id': query_id[i].item(),
                        'label': label[i].item(),
                        'bucket': bucket[i].item(),
                        'fused_scores': fused_scores[i].item(),
                    })

                    fused_evaluation_scores.flush()
                    os.fsync(fused_evaluation_scores.fileno())


def evaluate_ranking_metrics_multi_task(path=f'{Config.EVALUATION_PATH.value}/{Config.MULTI_TASK_FUSE_EVALUATION.value}'):
    df = pd.read_csv(path)
    results = {metric: [] for metric in REL_METRICS_CUSTOM}

    for query_id, group in df.groupby(by='query_id'):
        sorted_group = group.sort_values(by='fused_scores', ascending=False)

        labels = sorted_group['label'].values
        scores = sorted_group['fused_scores'].values

        for name, metric_function in REL_METRICS_CUSTOM.items():
            metric_value = float(metric_function(scores, labels))
            results[name].append(metric_value)

    average_results = {name: sum(metric_scores) / len(metric_scores) for name, metric_scores in results.items()}

    return average_results


def evaluate_ranking_metrics_clicks(path=f'{Config.EVALUATION_PATH.value}/{Config.CLICK_EVALUATION.value}'):
    df = pd.read_csv(path)
    results = {metric: [] for metric in REL_METRICS_CUSTOM}

    for query_id, group in df.groupby(by='query_id'):
        sorted_group = group.sort_values(by='click_scores', ascending=False)

        labels = sorted_group['label'].values
        scores = sorted_group['click_scores'].values

        for name, metric_function in REL_METRICS_CUSTOM.items():
            metric_value = float(metric_function(scores, labels))
            results[name].append(metric_value)

    average_results = {name: sum(metric_scores) / len(metric_scores) for name, metric_scores in results.items()}

    return average_results


if __name__ == '__main__':
    set_seed()

    evaluation_df = load_evaluation_scores()
    train_df, test_df = create_train_test_split(evaluation_df=evaluation_df, test_size=0.7)
    features = ['click_scores', 'skip_scores', 'dwell_time_scores']
    label = 'label'
    
    train_dataset = MultiTaskFusionDataset(
        df=train_df,
        features=features,
        label=label,
    )

    test_dataset = MultiTaskFusionDataset(
        df=test_df,
        features=features,
        label=label,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=TEST_BATCH_SIZE,
    )

    model = MultiTaskFusionModel()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    if IS_MULTI_TASK:
        if USE_KFOLD:
            train_multi_task_fuse_kfold(
                model=model,
                dataset=train_dataset,
                df=train_df,
                kfolds=KFolds,
                optimizer=optimizer,
                criterion=criterion,
                epochs=EPOCHS,
            )

        else:
            train_multi_task_fuse(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=EPOCHS,
            )

        evaluate_multi_task_fuse(
            model=model,
            dataloader=test_dataloader,
        )

        evaluate_ranking_metrics_multi_task()

    else:
        evaluate_ranking_metrics_clicks()

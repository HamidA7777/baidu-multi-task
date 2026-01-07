import torch
import csv
import os
from config import Config


def evaluate(model, dataloader, save_path=f'{Config.EVALUATION_PATH.value}/{Config.MULTI_TASK_EVALUATION.value}'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', newline='') as evaluation_scores:
        if model.multi_task:
            fieldnames = ['query_id', 'label', 'bucket', 'click_scores', 'skip_scores', 'dwell_time_scores']
        else:
            fieldnames = ['query_id', 'label', 'bucket', 'click_scores']

        writer = csv.DictWriter(evaluation_scores, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for batch in dataloader:
                batch['tokens'] = batch['tokens'].to(device)
                batch['attention_mask'] = batch['attention_mask'].to(device)
                batch['token_types'] = batch['token_types'].to(device)
                
                if model.multi_task:
                    click_scores, skip_scores, dwell_time_scores = model.predict(**batch)

                    for i in range(len(click_scores)):
                        writer.writerow({
                            'query_id': batch['query_id'][i].item(),
                            'label': batch['label'][i].item(),
                            'bucket': batch['bucket'][i].item(),
                            'click_scores': click_scores[i],
                            'skip_scores': skip_scores[i],
                            'dwell_time_scores': dwell_time_scores[i],
                        })

                        evaluation_scores.flush()
                        os.fsync(evaluation_scores.fileno())

                else:
                    click_scores = model.predict(**batch)

                    for i in range(len(click_scores)):
                        writer.writerow({
                            'query_id': batch['query_id'][i].item(),
                            'label': batch['label'][i].item(),
                            'bucket': batch['bucket'][i].item(),
                            'click_scores': click_scores[i],
                        })

                        evaluation_scores.flush()
                        os.fsync(evaluation_scores.fileno())


if __name__ == '__main__':
    from data import EvaluationDataset
    from torch.utils.data import DataLoader
    from model import CrossEncoder
    from pathlib import Path
    from transformers import BertConfig
    from train_ddp import load_checkpoint
    from const import *

    PATH = f'{Config.CHECKPOINT_PATH.value}/{Config.MULTI_TASK_MODEL.value}'
    checkpoint = load_checkpoint(PATH)

    BATCH_SIZE = 1

    # hyperparameters for BERT Cross-Encoder
    VOCAB_SIZE = 22000
    HIDDEN_SIZE = 768
    NUM_HIDDEN_LAYERS = 12
    NUM_ATTENTION_HEADS = 12

    files = list(Path(Config.TEST_PATH.value).glob("*.txt"))

    max_sequence_length = MAX_SEQUENCE_LENGTH
    special_tokens = SPECIAL_TOKENS
    segment_types = SEGMENT_TYPES
    ignored_titles = [MISSING_TITLE, WHAT_OTHER_PEOPLE_SEARCHED_TITLE]

    test_dataset = EvaluationDataset(
        files=files,
        max_sequence_length=max_sequence_length,
        special_tokens=special_tokens,
        segment_types=segment_types,
        ignored_titles=ignored_titles,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
    )

    config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
    )

    model = CrossEncoder(config=config)

    model.load_state_dict(checkpoint['model'])

    evaluate(
        model=model,
        dataloader=test_dataloader,
    )


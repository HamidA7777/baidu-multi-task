import torch
from config import Config
import os


def estimate_bias_statistics(dataloader, max_docs=10000000):
    K = 10

    impressions = torch.zeros(K + 1, dtype=torch.long)
    clicks = torch.zeros(K + 1, dtype=torch.long)

    examined = torch.zeros(K + 1, dtype=torch.long)
    skips = torch.zeros(K + 1, dtype=torch.long)

    dwell_sum = torch.zeros(K + 1)
    dwell_count = torch.zeros(K + 1)

    doc_counter = 0

    for batch in dataloader:
        positions = batch['positions']
        click = batch['clicks']
        skip = batch['skips']
        dwell_time = batch['dwell_time']

        for k in range(1, K + 1):
            mask = (positions == k)

            impressions[k] += mask.sum()

            if click is not None:
                clicks[k] += click[mask].sum().long()

            if skip is not None:
                skips[k] += skip[mask].sum().long()
                examined[k] += mask.sum()

            if dwell_time is not None:
                dwell_sum[k] += dwell_time[mask].sum()
                dwell_count[k] += mask.sum()

        doc_counter += len(positions)

        if doc_counter >= max_docs:
            break

    results = {
        "docs_processed": doc_counter,
        "clicks": clicks.cpu().tolist(),
        "impressions": impressions.cpu().tolist(),
        "skips": skips.cpu().tolist(),
        "examined": examined.cpu().tolist(),
        "dwell_sum": dwell_sum.cpu().tolist(),
        "dwell_count": dwell_count.cpu().tolist(),
    }

    return results


def main():
    from data import CrossEncoderPretrainDataset
    from torch.utils.data import DataLoader
    from pathlib import Path
    from const import (
        MAX_SEQUENCE_LENGTH,
        MASKING_RATE,
        SPECIAL_TOKENS,
        SEGMENT_TYPES,
        MISSING_TITLE,
        WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
    )
    import json

    # hyperparameters for training
    BATCH_SIZE = 64

    files = list(Path(Config.TRAIN_PATH.value).glob("*.gz"))
    max_sequence_length = MAX_SEQUENCE_LENGTH
    masking_rate = MASKING_RATE
    mask_query = False
    mask_doc = True
    special_tokens = SPECIAL_TOKENS
    segment_types = SEGMENT_TYPES
    ignored_titles = [MISSING_TITLE, WHAT_OTHER_PEOPLE_SEARCHED_TITLE]

    train_dataset = CrossEncoderPretrainDataset(
        files=files,
        max_sequence_length=max_sequence_length,
        masking_rate=masking_rate,
        mask_query=mask_query,
        mask_doc=mask_doc,
        special_tokens=special_tokens,
        segment_types=segment_types,
        ignored_titles=ignored_titles,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )

    results = estimate_bias_statistics(train_dataloader)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, '..', 'utils', 'ultr_feedback_stats.json')

    with open(file_path, "w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()

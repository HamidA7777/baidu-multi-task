import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from config import Config
import numpy as np
import torch.nn.functional as F


def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint


def ddp_setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def train(model, dataloader, optimizer, rank, local_rank, save_path=f'{Config.CHECKPOINT_PATH.value}/{Config.MULTI_TASK_MODEL.value}'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device = torch.device(f'cuda:{local_rank}')
    model.to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model.train()

    total_loss = 0.0
    num_batches = 0
    steps = 0

    for batch in dataloader:
        batch['tokens'] = batch['tokens'].to(device)
        batch['attention_mask'] = batch['attention_mask'].to(device)
        batch['token_types'] = batch['token_types'].to(device)
        batch['labels'] = batch['labels'].to(device)
        batch['clicks'] = batch['clicks'].to(device).float() if batch['clicks'] is not None else None
        batch['skips'] = batch['skips'].to(device).float() if batch['skips'] is not None else None
        batch['dwell_time'] = batch['dwell_time'].to(device).float() if batch['dwell_time'] is not None else None
        batch['positions'] = batch['positions'].to(device)

        optimizer.zero_grad()
        loss_per_task, _ = model(**batch)

        click_loss = loss_per_task['click']
        skip_loss = loss_per_task.get('skip', 0.0)
        dwell_time_loss = loss_per_task.get('dwell_time', 0.0)
        mlm_loss = loss_per_task.get('mlm', 0.0)

        total_loss_step = (
            click_loss
            + skip_loss
            + dwell_time_loss
            + mlm_loss
        )

        total_loss_step.backward()
        optimizer.step()

        total_loss += total_loss_step.item()
        num_batches += 1
        steps += 1

        avg_loss = total_loss / num_batches

        if (rank == 0) and (steps % 1000 == 0):
            save_checkpoint(model.module, optimizer, save_path)

        if (steps == 2000000):
            # training is finished
            break


def main():
    rank, local_rank, world_size = ddp_setup()

    set_seed(1 + rank)

    from data import CrossEncoderPretrainDataset
    from model import CrossEncoder
    from torch.utils.data import DataLoader
    from pathlib import Path
    from transformers import BertConfig
    from const import (
        MAX_SEQUENCE_LENGTH,
        MASKING_RATE,
        SPECIAL_TOKENS,
        SEGMENT_TYPES,
        MISSING_TITLE,
        WHAT_OTHER_PEOPLE_SEARCHED_TITLE,
    )

    # hyperparameters for training
    BATCH_SIZE = 64

    # hyperparameters for optimizer
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01

    # hyperparameters for BERT Cross-Encoder
    VOCAB_SIZE = 22000
    HIDDEN_SIZE = 768
    NUM_HIDDEN_LAYERS = 12
    NUM_ATTENTION_HEADS = 12

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
        rank=rank,
        world_size=world_size,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )

    config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
    )

    model = CrossEncoder(config=config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    
    train(model=model, dataloader=train_dataloader, optimizer=optimizer, rank=rank, local_rank=local_rank)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()

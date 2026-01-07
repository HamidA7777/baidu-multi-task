from typing import Optional

import torch
from torch import LongTensor, FloatTensor, BoolTensor, IntTensor
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, BertForPreTraining, BertConfig
from transformers.models.bert.modeling_bert import BertLMPredictionHead


class BertModel(BertForPreTraining):
    """
    Basic BERT model pre-trained only on the MLM task (i.e. RoBERTa setup).
    The model can be further fine-tuned in a CrossEncoder or Condenser setup.
    """

    def __init__(self, config: PretrainedConfig):
        super(BertModel, self).__init__(config)
        self.mlm_head = BertLMPredictionHead(config)
        self.mlm_loss = CrossEntropyLoss()

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        **kwargs,
    ):
        loss = 0
        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            return_dict=True,
        )
        query_document_embedding = outputs.pooler_output

        if labels is not None:
            loss += self.get_mlm_loss(outputs[0], labels)

        return loss, query_document_embedding

    def get_mlm_loss(self, sequence_output: FloatTensor, labels: LongTensor):
        token_scores = self.mlm_head(sequence_output)

        return self.mlm_loss(
            token_scores.view(-1, self.config.vocab_size),
            labels.view(-1),
        )


class CrossEncoder(BertModel):
    """
    BERT cross-encoder: https://arxiv.org/abs/1910.14424
    Query and document are concatenated in the input. The prediction targets are an MLM
    task and a relevance prediction task using the CLS token. To reproduce the original
    model released by Baidu, we use clicks or annotations as the relevance signal.
    """

    def __init__(self, config: PretrainedConfig):
        super(CrossEncoder, self).__init__(config)
        self.click_head = nn.Linear(config.hidden_size, 1)
        self.skip_head = nn.Linear(config.hidden_size, 1)
        self.dwell_time_head = nn.Linear(config.hidden_size, 1)

        self.click_loss = nn.BCEWithLogitsLoss()
        self.skip_loss = nn.BCEWithLogitsLoss()
        self.dwell_time_loss = nn.MSELoss()

        # bias tower
        separate_logit_for_each_head = True
        position_vocab_size = 30
        d_model = 1
        num_logits = 3 if separate_logit_for_each_head else 1
        self.bias_tower = nn.Sequential(
            nn.Embedding(position_vocab_size, d_model),
            nn.Linear(d_model, num_logits),
        )

        self.multi_task = True
        self.debiasing = True

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        clicks: Optional[FloatTensor] = None,
        skips: Optional[FloatTensor] = None,
        dwell_time: Optional[FloatTensor] = None,
        positions: Optional[FloatTensor] = None,
        **kwargs,
    ):
        loss_per_task = {}

        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            return_dict=True,
        )
        query_document_embedding = outputs.pooler_output

        bias_scores = None
        
        if positions is not None:
            bias_scores = self.bias_tower(positions)

        # added this to optionally enable/disable debiasing
        if bias_scores is not None and not self.debiasing:
            bias_scores = torch.zeros_like(bias_scores)

        if clicks is not None:
            if bias_scores.size(1) == 1:
                click_scores = self.click_head(query_document_embedding) + bias_scores
            else:
                click_scores = self.click_head(query_document_embedding) + bias_scores[:,0].unsqueeze(-1)

            click_loss = self.click_loss(click_scores.squeeze(-1), clicks)
            loss_per_task['click'] = click_loss

        if self.multi_task:
            if skips is not None:
                if bias_scores.size(1) == 1:
                    skip_scores = self.skip_head(query_document_embedding) + bias_scores
                else:
                    skip_scores = self.skip_head(query_document_embedding) + bias_scores[:,1].unsqueeze(-1)

                skip_loss = self.skip_loss(skip_scores.squeeze(-1), skips)
                loss_per_task['skip'] = skip_loss

            if dwell_time is not None:
                if bias_scores.size(1) == 1:
                    dwell_time_scores = self.dwell_time_head(query_document_embedding) + bias_scores
                else:
                    dwell_time_scores = self.dwell_time_head(query_document_embedding) + bias_scores[:,2].unsqueeze(-1)

                dwell_time_loss = self.dwell_time_loss(dwell_time_scores.squeeze(-1), dwell_time)
                loss_per_task['dwell_time'] = dwell_time_loss

        if labels is not None:
            mlm_loss = self.get_mlm_loss(outputs[0], labels)
            loss_per_task['mlm'] = mlm_loss

        return loss_per_task, query_document_embedding

    def predict(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        **kwargs
    ):
        _, query_document_embedding = self.forward(
            tokens=tokens,
            attention_mask=attention_mask,
            token_types=token_types,
        )

        click_scores = self.click_head(query_document_embedding).squeeze(-1).cpu().tolist()
        skip_scores = self.skip_head(query_document_embedding).squeeze(-1).cpu().tolist()
        dwell_time_scores = self.dwell_time_head(query_document_embedding).squeeze(-1).cpu().tolist()

        if self.multi_task:
            return (click_scores, skip_scores, dwell_time_scores)

        return click_scores


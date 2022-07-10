from typing import Optional, Union, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import transformers
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .modeling_bert import BertModel
from .modified_bert import CustomizedPooler as BertPooler


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class OurBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls_type = config.cls_type
        if config.cls_type == "fc":
            self.self_pooler = BertPooler(config)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        elif config.cls_type == "cnn":
            n_filters = 256
            filter_sizes = [2, 3, 5]
            self.convs = Conv1d(config.hidden_size, n_filters, filter_sizes)
            self.classifier = nn.Linear(len(filter_sizes) * n_filters, config.num_labels)
        elif config.cls_type == "attention":
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.W_w = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
            self.u_w = nn.Parameter(torch.Tensor(config.hidden_size, 1))

            nn.init.uniform_(self.W_w, -0.1, 0.1)
            nn.init.uniform_(self.u_w, -0.1, 0.1)

        # Initialize weights and apply final processing
        self.post_init()

        # Freeze parts of pretrained model
        # config['freeze'] can be "all" to freeze all layers,
        # or any number of prefixes, e.g. ['embeddings', 'encoder']
        if config.freeze != "":
            for name, param in self.bert.named_parameters():
                if config.freeze == 'all' or name.startswith(config.freeze):
                    param.requires_grad = False
                    # print(f"Froze layer: {name}")

        # freeze_layers is a string "1,2,3" representing layer number
        if config.freeze_layers != "":
            layer_indexes = [int(x) for x in config.freeze_layers.split(",")]
            for layer_idx in layer_indexes:
                for param in list(self.bert.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
                # print(f"Froze Layer: {layer_idx}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        if self.cls_type == "cnn":
            encoded_layers = outputs[0].permute(0, 2, 1)
            # encoded_layers: [batch_size, bert_dim=768, seq_len]

            conved = self.convs(encoded_layers)
            # conved 是一个列表， conved[0]: [batch_size, filter_num, *]

            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                      for conv in conved]
            # pooled 是一个列表， pooled[0]: [batch_size, filter_num]

            cat = self.dropout(torch.cat(pooled, dim=1))
            # cat: [batch_size, filter_num * len(filter_sizes)]

            logits = self.classifier(cat)
        elif self.cls_type =="fc":
            pooled_output = self.self_pooler(attention_mask, outputs)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        elif self.cls_type == "attention":
            encoded_layers = self.dropout(outputs[0])
            # encoded_layers: [batch_size, seq_len, bert_dim=768]

            score = torch.tanh(torch.matmul(encoded_layers, self.W_w))
            # score: [batch_size, seq_len, bert_dim]

            attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
            # attention_weights: [batch_size, seq_len, 1]

            scored_x = encoded_layers * attention_weights
            # scored_x : [batch_size, seq_len, bert_dim]

            feat = torch.sum(scored_x, dim=1)
            # feat: [batch_size, bert_dim=768]
            logits = self.classifier(feat)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
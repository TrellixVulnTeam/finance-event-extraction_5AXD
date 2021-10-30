from __future__ import unicode_literals
from __future__ import absolute_import

from django.shortcuts import render
import myapp.models as models
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel

# 插入函数
from transformers import BertConfig, BertTokenizer


def insert(request):
    if request.method == "POST":
        username = request.POST.get("username", None)
        description = request.POST.get("description", None)

        MODEL_PATH = "C:/Users/David Xu/PycharmProjects/DLdemo/finEE/myapp/storage_model"
        config = BertConfig.from_pretrained(MODEL_PATH)
        config.output_hidden_states = True
        config.output_attentions = True
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        dmbert = DMBERT.from_pretrained(MODEL_PATH, config=config)

        raw_ids = tokenizer.encode(username)
        input_ids = raw_ids + [0 for _ in range(128 - len(raw_ids))]
        output_ids = set()
        for i in range(len(raw_ids)):
            idx = i + 1
            maskL = [[1.0 for _ in range(idx)] + [0.0 for _ in range(128 - idx)]]
            maskR = [[0.0 for _ in range(idx)] + [1.0 for _ in range(len(raw_ids) - idx)] + [0.0 for _ in
                                                                                             range(128 - len(raw_ids))]]
            sen_code = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [[1 for _ in range(len(raw_ids))] + [0 for _ in range(128 - len(raw_ids))]],
                    dtype=torch.long),
                "token_type_ids": torch.tensor([0 for _ in range(128)]),
                "maskL": torch.tensor(maskL, dtype=torch.float),
                "maskR": torch.tensor(maskR, dtype=torch.float),
                "label": torch.tensor([0], dtype=torch.long)
            }

            dmbert.eval()
            with torch.no_grad():
                inputs = {
                    "input_ids": sen_code["input_ids"],
                    "attention_mask": sen_code["attention_mask"],
                    "token_type_ids": sen_code["token_type_ids"],
                    "maskL": sen_code["maskL"],
                    "maskR": sen_code["maskR"],
                    "labels": sen_code["label"],
                }
                outputs = dmbert(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                output_ids.add(preds[0])

        twz = models.message.objects.create(sentence=username, description=description,
                                            predict_result=ids2labels(output_ids))
        twz.save()
    return render(request, '../templates/insert.html')


# 定义展示函数
def result_list(request):
    people_list = models.message.objects.all()
    return render(request, '../templates/show.html', {"people_list": people_list})


class DMBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.maxpooling = nn.MaxPool1d(128)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, maskL=None, maskR=None, labels=None):
        batchSize = input_ids.size(0)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        conved = outputs[0]
        conved = conved.transpose(1, 2)
        conved = conved.transpose(0, 1)
        L = (conved * maskL).transpose(0, 1)
        R = (conved * maskR).transpose(0, 1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(batchSize, self.config.hidden_size)
        pooledR = self.maxpooling(R).contiguous().view(batchSize, self.config.hidden_size)
        pooled = torch.cat((pooledL, pooledR), 1)
        pooled = pooled - torch.ones_like(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        reshaped_logits = logits.view(-1, self.config.num_labels)
        outputs = (reshaped_logits,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs
        return outputs


def ids2labels(ids):
    labels = ['None', 'Expense', 'Dividend', 'Rating', 'Profit-Loss', 'Financing', 'CSR-Brand', 'Product-Service',
              'Employment', 'Deal', 'Revenue', 'Merger-Acquisition', 'Legal', 'Investment', 'FinancialReport',
              'SalesVolume', 'SecurityValue', 'Facility', 'Macroeconomics']
    return [labels[id] for id in ids]

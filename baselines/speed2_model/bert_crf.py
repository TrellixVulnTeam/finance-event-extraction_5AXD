DEBUGGING=False

import logging
from transformers.modeling_bert import BertModel, BertPreTrainedModel
from transformers.configuration_bert import BertConfig
from torch.nn import CrossEntropyLoss

from crf import *
from utils_maven import to_crf_pad, unpad_crf

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    'bert-base-japanese': "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    'bert-base-japanese-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    'bert-base-japanese-char': "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    'bert-base-japanese-char-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    'bert-base-finnish-cased-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    'bert-base-finnish-uncased-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
}


class BertCRFForTokenClassification(BertPreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BertCRFForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_label_tokens = input("please input num_label_tokens, e.g. 248")
        self.num_types = input("please input num_label_tokens, e.g. 168")
        self.label_w = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.label_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.label_ffn = nn.Linear(self.num_types, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 2)
        self.crf = CRF(self.num_labels)

        self.init_weights()

    def _get_features(self, input_ids=None, attention_mask=None, token_type_ids=None,
                      position_ids=None, head_mask=None, inputs_embeds=None, **aux_inputs):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        if "_label_token_lens" in aux_inputs:
            label_token_embs, st_idx = [], 1 # 1 as to skip [CLS]
            for token_len in aux_inputs["_label_token_lens"]:
                if token_len == 1:
                    label_token_embs.append(sequence_output[:,st_idx,:])
                else:
                    assert token_len>0
                    label_token_embs.append(torch.sum(sequence_output[:, st_idx:st_idx+token_len, :], dim=1))
                st_idx = st_idx+token_len
            assert st_idx == self.num_label_tokens + 1
            assert len(label_token_embs) == self.num_types
            label_token_embs = torch.stack(label_token_embs, 1).transpose(1,2) # (batch_size, hidden_size, num_types)

            # (batch_size, len_cls_text_pad_tokens = max_seq_len, hidden_size)
            text_token_embs = torch.cat((sequence_output[:,0:1,:], sequence_output[:, st_idx+1:, :]), 1)
            if DEBUGGING: # debug printing
                print("text_token_embs (batch_size, max_seq_len, hidden_size):", text_token_embs.size())
                print("label_token_embs (batch_size, hidden_size, num_types):", label_token_embs.size())

            label_text_sims = torch.matmul(
                self.label_w(text_token_embs),
                label_token_embs
            ) # (batch_size, max_seq_len, num_types)
            aux_output = self.label_ffn(self.label_dropout(label_text_sims)) # (batch_size, max_seq_len, num_labels)

            sequence_output = text_token_embs

            if DEBUGGING: # debug printing
                print("label_text_sims (batch_size, max_seq_len, num_types):", label_text_sims.size())
                print("aux_output (batch_size, max_seq_len, num_labels):", aux_output.size())
                print(input("\n\ncontinue?"))
        else:
            print(aux_inputs)
            print(input("\n\ncontinue?"))

        feats = self.classifier(sequence_output)
        return feats, outputs, aux_output

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, pad_token_label_id=None, **aux_inputs):

        if "label_input_ids" in aux_inputs:
            label_input_ids = aux_inputs["label_input_ids"]
            label_input_mask = aux_inputs["label_input_mask"]
            label_segment_ids = aux_inputs["label_segment_ids"]

            assert self.num_label_tokens == len(label_input_ids) -2 # 2 as to exclude [CLS] and [SEP]

            if DEBUGGING:
                for _name, tensor_to_print in zip(["label_input_ids", "label_input_mask", "label_segment_ids", "text_input_ids", "attention_mask", "token_type_ids", "labels"],
                    [label_input_ids, label_input_mask, label_segment_ids, input_ids[0], attention_mask[0], token_type_ids[0], labels[0]]):
                    try:
                        print(_name, "({}) \t".format(len(tensor_to_print)), tensor_to_print)
                    except:
                        print(_name, tensor_to_print)
                print(input("\n\ncontinue?"))
            bs = input_ids.size()[0] # batch size
            input_ids = torch.cat((torch.stack(bs*[label_input_ids]), input_ids[:, 1:]), 1)
            token_type_ids = torch.cat((torch.stack(bs*[label_segment_ids]), attention_mask.detach().clone()[:, 1:]), 1)
            attention_mask_with_aux = torch.cat((torch.stack(bs*[label_input_mask]), attention_mask[:, 1:]), 1)
            if DEBUGGING:
                for _name, tensor_to_print in zip(["label_input_ids", "label_input_mask", "label_segment_ids", "text_input_ids", "attention_mask", "token_type_ids", "labels"],
                    [label_input_ids, label_input_mask, label_segment_ids, input_ids[0], attention_mask_with_aux[0], token_type_ids[0], labels[0]]):
                    try:
                        print(_name, "({}) \t".format(len(tensor_to_print)), tensor_to_print)
                    except:
                        print(_name, tensor_to_print)
                print(input("\n\ncontinue?"))

        logits, outputs, aux_output = self._get_features(input_ids, attention_mask_with_aux, token_type_ids,
                                             position_ids, head_mask, inputs_embeds, **aux_inputs)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            pad_mask = (labels != pad_token_label_id)

            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
                # active_labels = labels.view(-1)[active_loss]
                loss_mask = ((attention_mask == 1) & pad_mask)
            else:
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_mask = ((torch.ones(logits.shape) == 1) & pad_mask)

            crf_labels, crf_mask = to_crf_pad(labels, loss_mask, pad_token_label_id)
            crf_logits, _ = to_crf_pad(logits, loss_mask, pad_token_label_id)

            loss = self.crf.neg_log_likelihood(crf_logits, crf_mask, crf_labels)
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            best_path = self.crf(crf_logits, crf_mask)  # (torch.ones(logits.shape) == 1)
            best_path = unpad_crf(best_path, crf_mask, labels, pad_mask)

            # add contrastive loss
            cl_fn = CrossEntropyLoss()
            active_cl_mask = (labels.view(-1) != -100) & (labels.view(-1) !=0)
            active_logits = aux_output.view(-1, self.num_labels)[active_cl_mask]
            active_labels = labels.view(-1)[active_cl_mask]
            try:
                closs_weight = aux_inputs["closs_weight"] if "closs_weight" in aux_inputs else 1
                if len(active_labels) > 0:
                    closs = closs_weight * cl_fn(active_logits, active_labels)
                    #print("contrastive loss\t=", closs.item(), "\tcrf softmax loss\t=", loss.item())
                else:
                    closs = torch.tensor(0).cuda()
            except:
                print("active_logits", active_logits)
                print("active_labels", active_labels)
            loss += closs

            if True: # L1 regularization for weight label_ffn to force sparcity
                sparse_coef = 0.1
                sloss = sparse_coef * torch.sum(torch.abs(self.label_ffn.weight))
                loss += sloss
                # print("ffn L1 reg loss\t=", sloss.item())

            ###### if use contrastive prediction result as output #########
            if False:
                best_path = torch.argmax(aux_output, dim=2)
                best_path[pad_mask is False] = labels[pad_mask is False]
            #####################################################


            outputs = (loss,) + outputs + (best_path,)
        else:
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            if attention_mask is not None:
                mask = (attention_mask == 1)  # & (labels!=-100))
            else:
                mask = torch.ones(logits.shape).bool()  # (labels!=-100)
            crf_logits, crf_mask = to_crf_pad(logits, mask, pad_token_label_id)
            crf_mask = crf_mask.sum(axis=2) == crf_mask.shape[2]
            best_path = self.crf(crf_logits, crf_mask)
            temp_labels = torch.ones(mask.shape) * pad_token_label_id
            best_path = unpad_crf(best_path, crf_mask, temp_labels, mask)
            outputs = outputs + (best_path,)

        return outputs

import torch
import torch.nn as nn
from transformers import BertModel

class BertSegmenter(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 切割/非切割

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        logits = self.classifier(sequence_output)    # (batch, seq_len, 2)
        return logits 
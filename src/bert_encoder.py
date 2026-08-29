import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class BERTClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(BERTClassifier, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(cls_emb)
        return logits

    def extract_embeddings(self, text_list, max_length=128, device='cpu'):
        self.bert.to(device)
        inputs = self.tokenizer(text_list, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]
import torch
import torch.nn as nn

class GNNBERTFusion(nn.Module):
    def __init__(self, gnn_dim=64, bert_dim=768, hidden_dim=128, num_classes=10):
        super(GNNBERTFusion, self).__init__()
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.bert_proj = nn.Linear(bert_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, graph_emb, text_emb):
        g = self.gnn_proj(graph_emb)
        t = self.bert_proj(text_emb)
        g_unsq = g.unsqueeze(1)
        t_unsq = t.unsqueeze(1)
        attended_text, _ = self.cross_attention(query=g_unsq, key=t_unsq, value=t_unsq)
        attended_text = attended_text.squeeze(1)
        fused_z = torch.cat([g, attended_text], dim=-1)
        out = self.dropout(torch.relu(self.fc1(fused_z)))
        logits = self.fc2(out)
        return logits, fused_z
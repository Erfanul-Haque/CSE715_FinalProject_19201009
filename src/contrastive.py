"""Contrastive Dual-Encoder model for Cross-Modal Retrieval (Task 4)."""
import torch.nn as nn
import torch.nn.functional as F

class DualEncoder(nn.Module):
    """Projects GNN and BERT embeddings into a shared latent space for contrastive learning."""
    def __init__(self, gnn_dim=64, bert_dim=768, hidden_dim=128):
        super(DualEncoder, self).__init__()
        self.gnn_proj = nn.Linear(gnn_dim, hidden_dim)
        self.bert_proj = nn.Linear(bert_dim, hidden_dim)
        
    def forward(self, g, t):
        g = F.normalize(self.gnn_proj(g), p=2, dim=1)
        t = F.normalize(self.bert_proj(t), p=2, dim=1)
        return g, t
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GNNModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=10):
        super(GNNModel, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_embedding = global_mean_pool(x, batch)
        logits = self.fc1(graph_embedding)
        return logits, graph_embedding

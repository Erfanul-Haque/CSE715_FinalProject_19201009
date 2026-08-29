import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_segment_graph(segment_features, threshold=0.8):
    num_nodes = segment_features.shape[0]
    src_temporal = np.arange(num_nodes - 1)
    dst_temporal = np.arange(1, num_nodes)
    temporal_edges = np.vstack([
        np.stack([src_temporal, dst_temporal], axis=1),
        np.stack([dst_temporal, src_temporal], axis=1)
    ])
    feat_numpy = segment_features.numpy()
    sim_matrix = cosine_similarity(feat_numpy)
    sim_edges = np.argwhere((sim_matrix > threshold) & (sim_matrix < 1.0))
    all_edges = np.vstack([temporal_edges, sim_edges])
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    graph_data = Data(x=segment_features, edge_index=edge_index)
    return graph_data
"""Evaluation script for GNN and Fusion models (Tasks 2 & 3)."""
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader
from .gnn_model import GNNModel
import os

def evaluate_gnn():
    """Loads the trained GNN and evaluates it on the test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_path = os.path.join('..', 'data', 'processed', 'gtzan_graphs.pt')
    if not os.path.exists(data_path):
        print("Error: gtzan_graphs.pt not found.")
        return
        
    graph_list = torch.load(data_path, weights_only=False)
    
    # Split data (same split as training)
    from sklearn.model_selection import train_test_split
    _, test_graphs = train_test_split(graph_list, test_size=0.2, random_state=42)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    # Load model
    model = GNNModel(input_dim=128, hidden_dim=64, output_dim=10).to(device)
    model_path = os.path.join('..', 'data', 'processed', 'trained_gnn_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained GNN model.")
    else:
        print("Warning: trained_gnn_model.pth not found. Evaluating untrained model.")
        
    model.eval()
    preds = []
    truths = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            truths.extend(batch.y.cpu().tolist())
            
    # Calculate Metrics
    acc = accuracy_score(truths, preds)
    macro_f1 = f1_score(truths, preds, average='macro')
    micro_f1 = f1_score(truths, preds, average='micro')
    
    print("\n--- GNN Evaluation Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")

if __name__ == "__main__":
    evaluate_gnn()
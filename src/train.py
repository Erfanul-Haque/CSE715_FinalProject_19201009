"""Main training script for the GraphSAGE GNN model (Task 2)."""
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from .gnn_model import GNNModel
import os

def train_gnn():
    """Loads preprocessed graphs, splits data, and trains the GNN."""
    # Setup device (Uses GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the preprocessed graphs
    # Make sure you have downloaded gtzan_graphs.pt into data/processed/
    data_path = os.path.join('..', 'data', 'processed', 'gtzan_graphs.pt')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please download 'gtzan_graphs.pt' from Google Drive and place it in the 'data/processed/' folder.")
        return

    print("Loading preprocessed graphs...")
    graph_list = torch.load(data_path, weights_only=False)

    # 2. Split data into 80% Train and 20% Test
    train_graphs, test_graphs = train_test_split(graph_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # 3. Initialize Model, Optimizer, and Loss Function
    model = GNNModel(input_dim=128, hidden_dim=64, output_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Training on {len(train_graphs)} graphs, Testing on {len(test_graphs)} graphs.")
    print("Starting GNN Training...\n")

    # 4. Training Loop
    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            
        train_acc = correct / len(train_loader.dataset)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader.dataset):.4f} | Train Accuracy: {train_acc:.4f}")

    print("\nGNN Training Complete!")
    
    # 5. Save the trained model
    save_path = os.path.join('..', 'data', 'processed', 'trained_gnn_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_gnn()
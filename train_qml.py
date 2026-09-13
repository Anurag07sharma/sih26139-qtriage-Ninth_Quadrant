import os
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. PennyLane Quantum Circuit
n_qubits = 3
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc_circuit(inputs, weights):
    # Step 3: AngleEmbedding on 3 qubits
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # Entangling layers with parameterized rotations (RX, RY, RZ) and CNOTs
    for layer in weights:
        for i in range(n_qubits):
            qml.RX(layer[i, 0], wires=i)
            qml.RY(layer[i, 1], wires=i)
            qml.RZ(layer[i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
        
    return qml.expval(qml.PauliZ(0))

# 2. PyTorch Hybrid Model Wrapper
class HybridVQC(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer: projects 8 padded dimensions to 3 rotation angles for the 3 qubits
        self.encoder = nn.Linear(8, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qml_layer = qml.qnn.TorchLayer(vqc_circuit, weight_shapes)
        self.classifier = nn.Linear(1, 1)

    def forward(self, x):
        angles = torch.pi * torch.sigmoid(self.encoder(x))
        q_out = self.qml_layer(angles)
        if len(q_out.shape) == 1:
            q_out = q_out.unsqueeze(1)
        logits = self.classifier(q_out)
        return torch.sigmoid(logits)

def main():
    os.makedirs("models", exist_ok=True)
    
    print("Loading pre-encoded quantum data...")
    X_raw = np.load("data/quantum/heart_angle.npy")       # (300, 6)
    y_raw = np.load("data/quantum/heart_labels.npy")      # (300,)
    
    # Pad 6 features with 2 zeros to reach 8 dimensions -> 3 qubits
    X_padded = np.pad(X_raw, ((0, 0), (0, 2)), mode='constant', constant_values=0.0)
    print(f"X shape after 8-dim padding: {X_padded.shape}, y shape: {y_raw.shape}")
    
    X_t = torch.tensor(X_padded, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_raw, dtype=torch.float32).unsqueeze(1).to(device)
    
    model = HybridVQC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCELoss()
    
    epochs = 15
    batch_size = 32
    
    print(f"\n--- Training VQC for {epochs} Epochs ---")
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(X_t.size()[0])
        total_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, X_t.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_t[indices], y_t[indices]
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            correct += ((preds >= 0.5).float() == batch_y).sum().item()
            total += batch_x.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2%}")
        
    torch.save(model.state_dict(), "models/heart_qml.pt")
    print("\nSUCCESS: VQC model weights saved to models/heart_qml.pt")

if __name__ == "__main__":
    main()

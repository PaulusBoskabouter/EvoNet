import torch
import torch.nn as nn
import numpy as np
from IPython.display import clear_output

def base_model_train(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001, device='cuda', patience=5, drop_neuron=False):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.train_loss.append(loss.item())

        # Validation and early stopping
        model.eval()
        X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            model.val_loss.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        clear_output(wait=True)

        if drop_neuron and model.hidden_size >= 5 and np.random.uniform(0.0, 1.0) >= 0.3 and epoch % 10 == 0:
            index = np.random.randint(0, model.hidden_size)
            model.drop_neuron(index)
            print(f"Droppping {index}")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.plot_performance(epochs)




def evo_model_train():
    ...
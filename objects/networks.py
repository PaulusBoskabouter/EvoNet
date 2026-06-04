import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Network(nn.Module):
    def __init__(self, hidden_size=100):
        super(Network, self).__init__()

        self.hidden_size = hidden_size

        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        ) 

        self.train_loss = []
        self.val_loss = []

    
    def forward(self, x):
        return self.network(x)

    

    def plot_performance(self, xlim:int = None):
        x = np.array([i + 1 for i in range(len(self.train_loss))])
        if xlim is not None:
            plt.xlim(1, xlim)
        

        plt.plot(x, self.train_loss, label="train loss")
        plt.plot(x, self.val_loss, label="val loss")
        plt.xlabel('epochs')
        plt.ylabel('MSE loss')
        plt.title(f'Network training performance for hidden_dim={self.hidden_size}')
        plt.legend()
        plt.show()


    def save(self, folder:Path=Path("base_models"), filename:str=None):
        if filename is None:
            filename = f"base_{self.hidden_size}.pt"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }, folder/filename)


    def load(self, filepath: Path = None, device='cpu'):
        if filepath is None:
            filepath = Path("base_models") / f"base_{self.hidden_size}.pt"
        checkpoint = torch.load(filepath, map_location=device)

        self.load_state_dict(checkpoint['state_dict'])
        self.train_loss = checkpoint['train_loss']
        self.val_loss = checkpoint['val_loss']



class EvoNet(Network):
    def __init__(self, hidden_size):
        super().__init__(hidden_size=hidden_size)
        

        self.W = self.state_dict()['network.0.weight']
        self.B = self.state_dict()['network.0.bias']
        self.A = self.state_dict()['network.2.weight']

        self.neuron_data



    def drop_neuron(self, index):
        # Store the current device
        device = next(self.parameters()).device

        # Move model to CPU for modification
        self.cpu()

        # Get current state_dict
        state_dict = self.state_dict()

        # Extract weights and biases
        W = state_dict['network.0.weight']  # Shape: (hidden_size, 1)
        B = state_dict['network.0.bias']    # Shape: (hidden_size,)
        A = state_dict['network.2.weight']  # Shape: (1, hidden_size)

        # Remove the neuron at `index`
        W_new = torch.cat([W[:index], W[index+1:]])  # Remove row
        B_new = torch.cat([B[:index], B[index+1:]])  # Remove element
        A_new = torch.cat([A[:, :index], A[:, index+1:]], dim=1)  # Remove column

        # Update hidden_size
        new_hidden_size = W_new.shape[0]

        # Reinitialize the network with new shapes
        self.network = nn.Sequential(
            nn.Linear(1, new_hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(new_hidden_size, 1, bias=True)
        )

        # Update state_dict with new weights and biases
        with torch.no_grad():
            self.network[0].weight.data = W_new
            self.network[0].bias.data = B_new
            self.network[2].weight.data = A_new

        # Update hidden_size attribute
        self.hidden_size = new_hidden_size

        # Move model back to the original device
        self.to(device)
    
    def calculate_neuron_data(self):
        bend = -(self.A * self.B.t()) / (self.A * self.W.t())

        coeff = A * W.t()

        neuron_data = torch.stack((bend, coeff), dim=1) 
        self.neuron_data = neuron_data
        

        # Example use of getting distances
        distances = torch.cdist(neuron_data, neuron_data, p=2)


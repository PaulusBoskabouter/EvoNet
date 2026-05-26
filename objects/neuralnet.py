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
            nn.Linear(hidden_size, 1)
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

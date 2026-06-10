import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy


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
        """
        Short function for plotting the training performance

        Args:
            xlim: manually set the xlim, as we have early stopping. Purely visual thing 
        """
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


    def save(self, folder:Path=Path("models"), filename:str=None):
        """
        Short function that allows saving of the current state

        Args:
            filepath: pretty self explanatory, refers to the directory
            filename: pretty self explanatory, refers to the actual filename with extention
        """
        if filename is None:
            filename = f"base_{self.hidden_size}.pt"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }, folder/filename)


    def load(self, folder:Path=Path("models"), filename:str=None, device='cpu'):
    # def load(self, filepath: Path = None, device='cpu'):
        """
        Short function that allows reloading of weights from a savefile.

        Args:
            filepath: pretty self explanatory, refers to the directory
            Device: loads instantly to the gpu/cpu  
        """
        checkpoint = torch.load(folder/filename, map_location=device)

        self.load_state_dict(checkpoint['state_dict'])
        self.train_loss = checkpoint['train_loss']
        self.val_loss = checkpoint['val_loss']



class EvoNet(Network):
    def __init__(self, hidden_size:int, device:str = 'cpu'):
        super().__init__(hidden_size=hidden_size)
        

        # Lower = better
        self.fitness = float('inf')

        # Calculate the neuron vectors
        self.calculate_neuron_data()


        # Historic data:
        self.size_history = [hidden_size]
        self.fitness_history = []

        self.device = device



    
    def initialize_network(self, A_new, W_new, B_new):
        """
        Short function for reinitializing the network, this is needed for if the network drops neurons.
        """

        # Update hidden_size attribute (must happen before reinitializing the network,
        # since initialize_network builds layers from self.hidden_size)
        self.hidden_size = W_new.shape[0]

        self.network = nn.Sequential(
                    nn.Linear(1, self.hidden_size, bias=True),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, 1, bias=False)
                )
        # Update state_dict with new weights and biases
        with torch.no_grad():
            self.network[0].weight.data = W_new
            self.network[0].bias.data = B_new
            self.network[2].weight.data = A_new

        # Refresh the cached weight matrices and neuron data so they stay in sync
        # with the (now smaller) network — otherwise a later order_by_bend() would
        # reindex/reimpose the stale, wrong-shaped pre-drop tensors.
        self.calculate_neuron_data()
        self.order_by_bend()

        # Move model back to the original device
        self.to(self.device)



    def drop_neuron(self, index):
        """
        Title says it all, this drops a hidden layer neuron.

        Args:
            index: the index of the neuron that gets dropped.
        """

        # Move model to CPU for modification
        self.cpu()

        # Get current state_dict
        A, W, B = self.get_weights()

        # Remove the neuron at `index`
        W_new = torch.cat([W[:index], W[index+1:]])  # Remove row
        B_new = torch.cat([B[:index], B[index+1:]])  # Remove element
        A_new = torch.cat([A[:, :index], A[:, index+1:]], dim=1)  # Remove column

        
        # Reinitialize the network with new shapes
        self.initialize_network(A_new, W_new, B_new)


    def calculate_neuron_data(self):
        """
        Calculates the neuron data, meaning the point of bend and coeff from that bend onwards. This is used for ordering and crossover. 
        """
        # Ensure A, B, W are 1D or 2D tensors for element-wise operations

        A, W, B = self.get_weights()

        A = A.squeeze()
        W = W.squeeze()
        B = B.squeeze()

        # Compute bend and coeff for each neuron
        bend = -B / W
        coeff = A * W

        # Stack along the last dimension to get shape (hidden_size, 2)
        neuron_data = torch.stack((bend, coeff), dim=1)
        self.neuron_data = neuron_data

    def order_by_bend(self):
        """
        See function above, based on the bend location along the x-axis we order the matrices. 
        """
        sorted_indices = torch.argsort(self.neuron_data[:, 0])  # Sort by bend (first column)
        
        self.neuron_data = self.neuron_data[sorted_indices]

        # Get current state_dict
        A, W, B = self.get_weights()
        # Extract weights and biases and reorder
        W = W[sorted_indices]
        B = B[sorted_indices]
        A = A[:, sorted_indices]


        with torch.no_grad():
            self.network[0].weight.data = W
            self.network[0].bias.data = B
            self.network[2].weight.data = A
    
    def get_split_index(self, x_value:float):
        """
        Used when the network is a parent and a child has to endure crossover. We determine what the index is of the weight matrices based on some x_value cut off. 

        args:
            x_value: the cut-off value.

        """
        for index, (bend, coeff) in enumerate(self.neuron_data):
            if bend >= x_value:
                return index


    def get_weights(self):
        # Get current state_dict
        state_dict = self.state_dict()

        # Extract weights and biases and reorder
        W = state_dict['network.0.weight']
        B = state_dict['network.0.bias']
        A = state_dict['network.2.weight']

        return A, W, B

    def agglomerate(self):
        # Find the eucledian closes mathematical neighbouring neuron
        distances = torch.cdist(self.neuron_data, self.neuron_data, p=2)
        shortest_distance = float('inf')
        best_i = None
        best_j = None
        for i in range(1, self.hidden_size):
            for j in range(0, i):
                d = distances[i][j]
                if d < shortest_distance:
                    shortest_distance = d
                    best_i = i
                    best_j = j


        # Extract values
        self.cpu()
        A, W, B = self.get_weights()
        w1, b1, a1 = W[best_i, :], B[best_i], A[:, best_i]
        w2, b2, a2 = W[best_j, :], B[best_j], A[:, best_j]

        # Calculate w
        w = a1 * w1 + a2 * w2

        # Calculate b
        b = (b1 * a1 + b2 * a2)
        
        # Adjust w & b, set a
        if w1 > 0 and w2 > 0:
            if w > 0:
                a = 1
            elif w < 0:
                w *= -1
                b *= -1
            a = -1
        elif w1 < 0 and w2 < 0:
            if w < 0:
                a = 1
            if w > 0:
                w *= -1
                b *= -1
                a = -1
        else:
            return # We don't agglomerate
        

        self.drop_neuron(best_i)
        self.drop_neuron(best_j)
        # Refetch weights
        A, W, B = self.get_weights()
        # Reshape for concatination
        a = torch.tensor(a).view(1,1)
        w = w.view(1, 1)

        A_new = torch.cat((A, a), dim=1) 
        B_new = torch.cat((B, b))
        W_new = torch.cat((W, w), dim=0)


        # Reinitialize with new params
        self.initialize_network(A_new, W_new, B_new)
        

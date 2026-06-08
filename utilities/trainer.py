import torch
import torch.nn as nn
import numpy as np
from objects.networks import Network, EvoNet
from IPython.display import clear_output
from copy import deepcopy

def base_model_train(model:Network, X_train:list, y_train:list, X_val:list, y_val:list, epochs:int = 100, lr:float = 0.001, device:str = 'cuda', patience:int = 15, plot:bool = True):
    """
    I think we're all familiar with a standard MLP training loop right?

    args:
        model: can be Network class model or EvoNet
        X_train: train data
        y_train: train labels
        x_val:   val data
        y_val:   val labels
        epochs:  yeah...
        lr:      yeah...
        device:  yeah...
        patience: for earlystopping
        plot:    boolean whether we should plot the performance.

    """
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
        
        
        # This insert statement exists for if we want to train without plotting the metrics each time. 
        if plot:
            clear_output(wait=True)
            model.plot_performance(epochs)






def evolving_trainer(populations:list, x_train:list, y_train:list, x_val:list, y_val:list, K:int=5, gens:int = 100, epochs:int = 5, crossover:float =0.1, mutation_rate:float =0.15, device: str='cpu') -> tuple[list, list]:
    """
    Trains a population of neural networks using an evolutionary algorithm with:
    - K-tournament selection for parent selection.
    - Crossover and mutation for reproduction.
    - Elitism to preserve the best network each generation.
    - Fitness calculation based on validation loss and hidden size penalty.

    The function iteratively evolves the population over `gens` generations, optimizing
    each network using `base_model_train` and evaluating fitness. The best networks
    from each generation are tracked in `generational_talent`.

    ---

    Args:
        populations (list[list[EvoNet]]):
            A list of lists, where `populations[generation_idx][net_idx]` is an `EvoNet` object.
            The last sublist (`populations[-1]`) is used as the starting population for evolution.

        x_train (torch.Tensor):
            Training input data of shape [N, input_features].

        y_train (torch.Tensor):
            Training target data of shape [N, output_features].

        x_val (torch.Tensor):
            Validation input data of shape [N, input_features].

        y_val (torch.Tensor):
            Validation target data of shape [N, output_features].

        K (int, optional):
            Number of networks sampled for K-tournament selection. Default: 5.

        gens (int, optional):
            Number of generations to evolve. Default: 100.

        epochs (int, optional):
            Number of training epochs per network per generation. Default: 5.

        crossover (float, optional):
            Probability of performing crossover (instead of cloning a parent). Default: 0.1.

        mutation_rate (float, optional):
            Probability of mutating a child network (e.g., dropping neurons). Default: 0.15.

        device (str, optional):
            Device to train on ('cpu' or 'cuda'). Default: 'cpu'.

    ---

    Returns:
        tuple[list[list[EvoNet]], list[EvoNet]]:
            - `populations`: Updated list of populations for all generations, including the initial population.
            - `generational_talent`: List of the best `EvoNet` from each generation (including generation 0).

    """
    generational_talent = [None] #  The None is a placeholder for the first gen
    population = populations[-1]
    
    best_fitness = float('inf')
    for net in population:
        base_model_train(net, x_train, y_train, x_val, y_val, epochs=epochs, lr=1e-3, device=device, patience=25, plot=False)
        net.fitness = net.val_loss[-1] + max(0, 0.05 * (net.hidden_size -7))
        if net.fitness < best_fitness:
            generational_talent[0] = net

        net.order_by_bend()


    for g in range(gens):
        clear_output(wait=True)
        print("Generation: ", g+1)
        population = populations[-1]
        new_generation = []

        # Do the elitism check
        best_fitness = float('inf')
        best_net = None
        for p in population:
            if p.fitness < best_fitness:
                best_net = p

        new_generation.append(best_net)
        generational_talent.append(best_net)

        while len(new_generation) < len(population):

            # Randomly select parents
            parents = [None, None]
            parent_fitness = [float('inf'), float('inf')]

            for p in range(2):
                temp = []
                for net in np.random.choice(population, size=K):
                    temp.append(net)             
                    if net.fitness < parent_fitness[p]:
                        if parents[0] != net: # Wouldn't want asexual reproduction
                            parents[p]= net
                            parent_fitness[p] = net.fitness
                if parents[p] is None:
                    print("FOUT CHECK DE LIJST")
                    print(parent_fitness)
                    return temp

            # Metrics, plotten. Hidden size per generatie, de fitness over generatie, 
            # Toevoegen neuronen?

            p1, p2 = parents 

            # We create a split based on bending point, between -0.5 and 0.5 
            split = np.random.uniform(-.5, .5) 
            p1_split = p1.get_split_index(split)
            p2_split = p2.get_split_index(split)

            # p1_split = p1.hidden_size //2 -1
            # p2_split = p2.hidden_size //2 -1


            mutation_rate = 0.1

            for child in range(2):
                # Crossover
                if np.random.uniform(0.0, 1.0) < crossover:
                    if child % 2 == 0:
                        A = torch.cat([p1.A[:, :p1_split], p2.A[:, p2_split:]], dim=1)
                        W = torch.cat([p1.W[:p1_split, :], p2.W[p2_split:, :]], dim=0)
                        B = torch.cat([p1.B[:p1_split], p2.B[p2_split:]], dim=0)
                    else:
                        A = torch.cat([p2.A[:, :p1_split], p1.A[:, p2_split:]], dim=1)
                        W = torch.cat([p2.W[:p1_split, :], p1.W[p2_split:, :]], dim=0)
                        B = torch.cat([p2.B[:p1_split], p1.B[p2_split:]], dim=0)
                    

                    try:
                        new_hidden_size = A.shape[1]
                        assert new_hidden_size>= 2
                        
                        c = EvoNet(hidden_size=new_hidden_size)
                        c.set_params(W, B, A)
                        
                    
                        
                    except AssertionError: # Else just copy parent
                        pass
                        c = deepcopy(parents[child])
                        c.val_loss = []
                        c.train_loss = []
                

                # Else copy parent
                else:
                    c = deepcopy(parents[child])
                    c.val_loss = []
                    c.train_loss = []
                    
                
                if np.random.uniform(0.0, 1.0) < mutation_rate:
                    count = np.random.randint(1, 4)
                    for _ in range(count):
                        neuron_index = np.random.randint(0, c.hidden_size)
                        c.drop_neuron(neuron_index)

                # Optimize new child
                base_model_train(c, x_train, y_train, x_val, y_val, epochs=epochs, lr=1e-3, device=device, patience=25, plot=False)
                c.fitness = np.random.randint(0, 10) # c.val_loss[-1] + max(0, 0.1 * (c.hidden_size -7))
                c.order_by_bend()

                new_generation.append(c)

                if len(new_generation) == len(population): # else can get too many kids
                    break


        populations.append(new_generation)

    best_fitness = float('inf')
    best_net = None
    for p in populations[-1]:
        if p.fitness < best_fitness:
            best_net = p
    generational_talent.append(best_net)
    

    return populations, generational_talent

    

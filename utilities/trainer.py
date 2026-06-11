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
                    break

        if plot:
            clear_output(wait=True)
            model.plot_performance(epochs)






def evolving_trainer(populations:list, x_train:list, y_train:list, x_val:list, y_val:list, K:int=5, gens:int = 100, epochs:int = 5, 
                     crossover:float =0.1, mutation_rate:float =0.15, device: str='cpu', es_patience:int=25, es_tol:float=1e-4, 
                     alpha:float=0.1, random:bool=False) -> tuple[list, list]:
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
        net.fitness = (1+ net.val_loss[-1]) * (1 + alpha * max(0, net.hidden_size - 7))
        if net.fitness < best_fitness:
            best_fitness = net.fitness
            generational_talent[0] = net

        net.order_by_bend()


    es_best = float('inf')
    es_no_improve = 0

    for g in range(gens):
        population = populations[-1]
        new_generation = []

        # Do the elitism check
        best_fitness = float('inf')
        best_net = None
        for p in population:
            if p.fitness < best_fitness:
                best_fitness = p.fitness
                best_net = p

        new_generation.append(best_net)
        generational_talent.append(best_net)

        while len(new_generation) < len(population):

            # Randomly select parents via K-tournament selection
            parents = [None, None]
            parent_fitness = [float('inf'), float('inf')]

            for p in range(2):
                for net in np.random.choice(population, size=K):
                    if p == 1 and net is parents[0]: # Wouldn't want asexual reproduction
                        continue
                    if net.fitness < parent_fitness[p]:
                        parents[p] = net
                        parent_fitness[p] = net.fitness
                if parents[p] is None:
                    # Fallback for small/homogeneous populations where every sampled
                    # net for the second parent matched the first parent
                    candidates = [net for net in population if net is not parents[0]] or population
                    parents[p] = candidates[np.random.randint(len(candidates))]

            p1, p2 = parents


            for child in range(2):
                # Crossover
                if np.random.uniform(0.0, 1.0) < crossover:
                    p1_A, p1_W, p1_B = p1.get_weights() 
                    p2_A, p2_W, p2_B = p2.get_weights() 
                    
                    if random: # Randomly neurons from both parents
                        random_1_select = np.random.choice(np.arange(p1.hidden_size), size=p1.hidden_size // 2, replace=False)
                        random_2_select = np.random.choice(np.arange(p2.hidden_size), size=p2.hidden_size // 2, replace=False)

                        A = torch.cat([p1_A[:, random_1_select], p2_A[:, random_2_select]], dim=1)
                        W = torch.cat([p1_W[random_1_select, :], p2_W[random_2_select, :]], dim=0)
                        B = torch.cat([p1_B[random_1_select], p2_B[random_2_select]], dim=0)

                    else: # Else use the split heuristic to determine the split
                        # We create a split based on bending point, between -0.5 and 0.5
                        split = np.random.uniform(-.5, .5)
                        p1_split = p1.get_split_index(split)
                        p2_split = p2.get_split_index(split)
                        if child % 2 == 0:
                            A = torch.cat([p1_A[:, :p1_split], p2_A[:, p2_split:]], dim=1)
                            W = torch.cat([p1_W[:p1_split, :], p2_W[p2_split:, :]], dim=0)
                            B = torch.cat([p1_B[:p1_split], p2_B[p2_split:]], dim=0)
                        else:
                            A = torch.cat([p2_A[:, :p1_split], p1_A[:, p2_split:]], dim=1)
                            W = torch.cat([p2_W[:p1_split, :], p1_W[p2_split:, :]], dim=0)
                            B = torch.cat([p2_B[:p1_split], p1_B[p2_split:]], dim=0)
                        
                    try:
                        new_hidden_size = A.shape[1]
                        assert new_hidden_size>= 2
                        
                        c = EvoNet(hidden_size=new_hidden_size)
                        c.initialize_network(A, W, B)
                        
                    except AssertionError:
                        pass
                

                # Else copy parent
                else:
                    c = deepcopy(parents[child])
                    c.val_loss = []
                    c.train_loss = []
                    
                
                if np.random.uniform(0.0, 1.0) < mutation_rate:
                    count = np.random.randint(1, 4)
                    for _ in range(count):
                        # NOTE I WAS BEING LAZY HERE, to re-enable agglomeration uncomment these lines and delete/comment the neuron 2 neuron dropping lines further below
                        # if c.hidden_size <= 4: # Keep at least 4 neurons, matching the crossover floor
                        #     break
                        # succes = c.agglomerate()
                        # if not succes: # Drop a neuron instead
                            # neuron_index = np.random.randint(0, c.hidden_size)
                            # c.drop_neuron(neuron_index)
                        
                        # NOTE refers to these two lines below
                        neuron_index = np.random.randint(0, c.hidden_size)
                        c.drop_neuron(neuron_index)

                # Optimize new child
                base_model_train(c, x_train, y_train, x_val, y_val, epochs=epochs, lr=1e-3, device=device, patience=30, plot=False)
                c.fitness = (1 + c.val_loss[-1]) * (1 + alpha * max(0, c.hidden_size - 7))
                c.order_by_bend()

                new_generation.append(c)

                if len(new_generation) == len(population): # else can get too many kids
                    break


        populations.append(new_generation)

        fitnesses = [n.fitness for n in new_generation]
        sizes = [n.hidden_size for n in new_generation]
        best_f = min(fitnesses)
        best_h = sizes[fitnesses.index(best_f)]
        print(
            f"Gen {g+1:>3}/{gens} | "
            f"best_fit={best_f:.5f}  avg_fit={sum(fitnesses)/len(fitnesses):.5f} | "
            f"best_h={best_h}  avg_h={sum(sizes)/len(sizes):.1f}",
            flush=True
        )

        if es_best - best_f > es_tol:
            es_best = best_f
            es_no_improve = 0
        else:
            es_no_improve += 1
            if es_no_improve >= es_patience:
                print(f"Early stopping at gen {g+1} (no improvement for {es_patience} gens)", flush=True)
                break

    best_fitness = float('inf')
    best_net = None
    for p in populations[-1]:
        if p.fitness < best_fitness:
            best_fitness = p.fitness
            best_net = p
    generational_talent.append(best_net)
    

    return populations, generational_talent

    

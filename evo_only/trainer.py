import torch
import torch.nn as nn
import numpy as np
from objects.networks import Network, EvoNet
from IPython.display import clear_output
from copy import deepcopy


def gaussian_perturb(model: EvoNet, std: float = 0.1):
    """Add Gaussian noise to all weights (W, A) and biases (B) in-place."""
    A, W, B = model.get_weights()
    with torch.no_grad():
        model.network[0].weight.data = W + torch.randn_like(W) * std
        model.network[0].bias.data   = B + torch.randn_like(B) * std
        model.network[2].weight.data = A + torch.randn_like(A) * std
    model.calculate_neuron_data()


def flip_vertical(model: EvoNet):
    """Negate A for a random neuron — turns a rising kink into falling or vice versa."""
    idx = np.random.randint(0, model.hidden_size)
    with torch.no_grad():
        model.network[2].weight.data[:, idx] *= -1
    model.calculate_neuron_data()


def flip_horizontal(model: EvoNet):
    """Negate W for a random neuron — reflects its bend position around x=0."""
    idx = np.random.randint(0, model.hidden_size)
    with torch.no_grad():
        model.network[0].weight.data[idx, :] *= -1
    model.calculate_neuron_data()


def nearest_neighbour_crossover(primary: EvoNet, secondary: EvoNet, alpha: float) -> EvoNet:
    """
    Blend each neuron in primary with its closest-bend-position match in secondary.
    Child has the same size as primary. alpha controls how much of primary is kept.
    """
    p_A, p_W, p_B = primary.get_weights()
    s_A, s_W, s_B = secondary.get_weights()

    primary_bends   = primary.neuron_data[:, 0]
    secondary_bends = secondary.neuron_data[:, 0]

    A_new = torch.zeros_like(p_A)
    W_new = torch.zeros_like(p_W)
    B_new = torch.zeros_like(p_B)

    for i in range(primary.hidden_size):
        j = torch.argmin(torch.abs(secondary_bends - primary_bends[i])).item()
        W_new[i]    = alpha * p_W[i]    + (1 - alpha) * s_W[j]
        B_new[i]    = alpha * p_B[i]    + (1 - alpha) * s_B[j]
        A_new[:, i] = alpha * p_A[:, i] + (1 - alpha) * s_A[:, j]

    c = EvoNet(hidden_size=primary.hidden_size)
    c.initialize_network(A_new, W_new, B_new)
    return c


def eval_only(model: EvoNet, X_val: list, y_val: list, device: str = 'cpu'):
    """Evaluate model fitness via a single forward pass — no gradient updates."""
    criterion = nn.MSELoss()
    model = model.to(device)
    model.eval()
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()
    model.val_loss.append(val_loss)


def evolving_trainer(populations: list, x_val: list, y_val: list, K: int = 5, gens: int = 100,
                     crossover: float = 0.1, mutation_rate: float = 0.15, perturb_std: float = 0.1,
                     device: str = 'cpu', es_patience: int = 25, es_tol: float = 1e-4) -> tuple[list, list]:
    """
    Evolves a population of neural networks using only the genetic algorithm (no local backpropagation).

    Fitness is evaluated via a single forward pass on the validation set. Selection, crossover,
    and mutation follow the same logic as evolving_trainer in utilities/trainer.py.

    Args:
        populations: list of lists; populations[-1] is used as the starting population.
        x_val:       validation inputs.
        y_val:       validation targets.
        K:           tournament size for parent selection.
        gens:        number of generations.
        crossover:   probability of crossover (vs. cloning a parent).
        mutation_rate: probability of mutating a child (neuron drop).
        device:      'cpu' or 'cuda'.
        es_patience: early-stopping patience in generations.
        es_tol:      minimum fitness improvement to reset the patience counter.

    Returns:
        (populations, generational_talent): all generations and per-generation best nets.
    """
    generational_talent = [None]
    population = populations[-1]

    best_fitness = float('inf')
    for net in population:
        eval_only(net, x_val, y_val, device=device)
        net.fitness = net.val_loss[-1] * (1 + 0.02 * max(0, net.hidden_size - 20))
        if net.fitness < best_fitness:
            best_fitness = net.fitness
            generational_talent[0] = net
        net.order_by_bend()

    es_best = float('inf')
    es_no_improve = 0

    for g in range(gens):
        population = populations[-1]
        new_generation = []

        best_fitness = float('inf')
        best_net = None
        for p in population:
            if p.fitness < best_fitness:
                best_fitness = p.fitness
                best_net = p

        new_generation.append(best_net)
        generational_talent.append(best_net)

        while len(new_generation) < len(population):

            parents = [None, None]
            parent_fitness = [float('inf'), float('inf')]

            for p in range(2):
                for net in np.random.choice(population, size=K):
                    if p == 1 and net is parents[0]:
                        continue
                    if net.fitness < parent_fitness[p]:
                        parents[p] = net
                        parent_fitness[p] = net.fitness
                if parents[p] is None:
                    candidates = [net for net in population if net is not parents[0]] or population
                    parents[p] = candidates[np.random.randint(len(candidates))]

            p1, p2 = parents

            for child in range(2):
                if np.random.uniform(0.0, 1.0) < crossover:
                    alpha = np.random.uniform(0.0, 1.0)
                    # child 0: primary=p1, child 1: primary=p2 (swap roles for variety)
                    if child % 2 == 0:
                        c = nearest_neighbour_crossover(p1, p2, alpha)
                    else:
                        c = nearest_neighbour_crossover(p2, p1, alpha)
                    c.val_loss = []
                    c.train_loss = []

                else:
                    parents[child].save(filename="temp.pt")
                    c = EvoNet(hidden_size=parents[child].hidden_size)
                    c.load(filename="temp.pt")
                    c.val_loss = []
                    c.train_loss = []

                if np.random.uniform(0.0, 1.0) < mutation_rate:
                    count = np.random.randint(1, 4)
                    for _ in range(count):
                        if c.hidden_size <= 2:
                            break
                        neuron_index = np.random.randint(0, c.hidden_size)
                        c.drop_neuron(neuron_index)

                if np.random.uniform(0.0, 1.0) < mutation_rate:
                    gaussian_perturb(c, std=perturb_std)

                if np.random.uniform(0.0, 1.0) < mutation_rate:
                    flip_vertical(c)

                if np.random.uniform(0.0, 1.0) < mutation_rate:
                    flip_horizontal(c)

                eval_only(c, x_val, y_val, device=device)
                c.fitness = c.val_loss[-1] * (1 + 0.02 * max(0, c.hidden_size - 20))
                c.order_by_bend()

                new_generation.append(c)

                if len(new_generation) == len(population):
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

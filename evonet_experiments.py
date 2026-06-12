import time
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from objects.Functions import SpikeFunc
from objects.networks import Network, EvoNet
from utilities.data_sampler import get_train_val_test


# evolving_trainer calls IPython's clear_output(wait=True), which is meant for
# notebook cells and just spams ANSI clear-screen escape codes to a plain log
# file/terminal. Neutralize it so the "Generation: N" progress prints cleanly.
import utilities.trainer as trainer_module
trainer_module.clear_output = lambda wait=False: None

# Prefix evolving_trainer's "Generation: N" prints with elapsed time and
# flush immediately, so progress is visible while tailing the log live.
_run_start = time.time()
_real_print = print
def _progress_print(*args, **kwargs):
    elapsed_min = (time.time() - _run_start) / 60
    kwargs["flush"] = True
    _real_print(f"[{elapsed_min:7.2f} min]", *args, **kwargs)
trainer_module.print = _progress_print

from utilities.trainer import evolving_trainer

results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

spike = SpikeFunc()
train, val = get_train_val_test(spike, 42, 10000)
x_train, y_train = train
x_val, y_val = val

######################################################################
#                   HYPER PARAMS                                     #
######################################################################


HYPERPARAMS = {
    "exp_id": 4,
    "alpha":  0.01,
    "population_size": 200,
    "initial_hiddem_dim": 150,
    "K": 5,
    "crossover_rate":0.2,
    "mutation_rate":0.1,
    "generations":300,
    "back_prop_epochs":150,
    "random_crossover": True,
    "random_mutation": True
}


P = [[EvoNet(hidden_size=HYPERPARAMS["initial_hiddem_dim"]) for _ in range(HYPERPARAMS["population_size"])]]

populations, best_networks = evolving_trainer(
    P, x_train, y_train, x_val, y_val,
    crossover=HYPERPARAMS['crossover_rate'], mutation_rate=HYPERPARAMS["mutation_rate"], K=HYPERPARAMS['K'], alpha=HYPERPARAMS['alpha'], 
    gens=HYPERPARAMS['generations'], epochs=HYPERPARAMS["back_prop_epochs"], device=device, es_patience=15, es_tol=1e-4, 
    random_crossover=HYPERPARAMS['random_crossover'], random_mutation=HYPERPARAMS['random_mutation']
)

print("DONE TRAINING", flush=True)

# Pick the true best net of the final population (fixing the same
# best_fitness-tracking bug found in trainer.py / notebook cell 9)
best_fitness = float('inf')
best_net = None
for net in populations[-1]:
    if net.fitness < best_fitness:
        best_fitness = net.fitness
        best_net = net

with open(results_folder / f"evonet_experiment_{HYPERPARAMS['exp_id']}.pkl", "wb") as f:
    pickle.dump({
        "Hyperparameters": HYPERPARAMS,
        "populations_summary": [
            {
                "fitness": [n.fitness for n in gen],
                "val_loss": [n.val_loss[-1] for n in gen],
                "hidden_size": [n.hidden_size for n in gen],
            }
            for gen in populations
        ],
        "generational_talent_fitness": [None if n is None else n.fitness for n in best_networks],
        "best_net_state": best_net.state_dict(),
        "best_net_hidden_size": best_net.hidden_size,
        "best_net_fitness": best_net.fitness,
    }, f)

# Disclaimer this is fully AI generated but it's all correct so yay.
# Prepare data for plotting
generations = len(populations)
fitness_mean = []
fitness_std = []
val_loss_mean = []
val_loss_std = []
hidden_sizes_mean = []
hidden_sizes_std = []

for gen in populations:
    fitness = [net.fitness for net in gen]
    val_loss = [net.val_loss[-1] for net in gen]
    hidden_size = [net.hidden_size for net in gen]

    fitness_mean.append(np.mean(fitness))
    fitness_std.append(np.std(fitness))
    val_loss_mean.append(np.mean(val_loss))
    val_loss_std.append(np.std(val_loss))
    hidden_sizes_mean.append(np.mean(hidden_size))  
    hidden_sizes_std.append(np.std(hidden_size))  

# Create a 1x3 subplot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Fitness (Mean ± Std)
ax1.errorbar(
    range(generations),
    fitness_mean,
    yerr=fitness_std,
    label="Fitness (Mean ± Std)",
    color="blue",
    capsize=5,
)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
ax1.set_title("Fitness Over Generations")
ax1.legend()
ax1.grid(True)

# Plot 2: Validation Loss (Mean ± Std)
ax2.errorbar(
    range(generations),
    val_loss_mean,
    yerr=val_loss_std,
    label="Val Loss (Mean ± Std)",
    color="orange",
    capsize=5,
)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Validation Loss")
ax2.set_title("Validation Loss Over Generations")
ax2.legend()
ax2.grid(True)

# Plot 3: Hidden Size
# ax3.plot(range(generations), hidden_sizes, marker="o", color="green")
ax3.errorbar(
    range(generations),
    hidden_sizes_mean,
    yerr=hidden_sizes_std,
    label="Val Loss (Mean ± Std)",
    color="green",
    capsize=5,
)
ax3.set_xlabel("Generation")
ax3.set_ylabel("Hidden Size")
ax3.set_title("Hidden Size Over Generations")
ax3.grid(True)

plt.tight_layout()
plt.savefig(results_folder/f"experiment_{HYPERPARAMS['exp_id']}_training.png")
# plt.show()o


best_fitness = float('inf')
best_net = None
gen = 0
for i, pop in enumerate(populations):
    for net in pop:
        if net.fitness < best_fitness:
            best_net = net
            gen = i
print(f"Best net found in generations {i+1}/{len(populations)}")

data = []
# net = np.random.choice(nets, size=1)[0]
X = np.arange(-spike.span * 1.25, spike.span * 1.25, 0.05) 
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
net = best_net.to(device)
net.eval()
Y = net(X).to(device)
X = X.detach().cpu().squeeze(1).numpy()
Y = Y.detach().cpu().squeeze(1).numpy()

data.append((X, Y, f"Gen {gen +1}"))


spike.plot(output_data=data, plot_title=f"Best EvoNet (hidden dimension: {net.hidden_size}, generation: {gen}) performance", save_path=results_folder/f"experiment_{HYPERPARAMS['exp_id']}_understanding.png")

best_net.save(filename=f"experiment_{HYPERPARAMS['exp_id']}.pt")

print(f"best_net hidden_size={best_net.hidden_size} fitness={best_net.fitness:.5f}", flush=True)


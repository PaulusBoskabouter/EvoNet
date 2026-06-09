import time
import pickle
import numpy as np
import torch
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

P = [[EvoNet(hidden_size=75) for _ in range(50)]]
populations, best_networks = evolving_trainer(
    P, x_train, y_train, x_val, y_val,
    crossover=0.15, mutation_rate=0.2, K=5, gens=300, epochs=150, device=device, es_patience=25, es_tol=1e-4
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

with open(results_folder / "evonet_run_a05_h75.pkl", "wb") as f:
    pickle.dump({
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

print(f"best_net hidden_size={best_net.hidden_size} fitness={best_net.fitness:.5f}", flush=True)
print("SAVED RESULTS to results/evonet_run_a05_h75.pkl", flush=True)

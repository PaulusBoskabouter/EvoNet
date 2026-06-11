import time
import pickle
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore", "FigureCanvasAgg is non-interactive")
from objects.Functions import SpikeFunc
from objects.networks import Network, EvoNet
from utilities.data_sampler import get_train_val_test

import evo_only.trainer as trainer_module
trainer_module.clear_output = lambda wait=False: None

_run_start = time.time()
_real_print = print
def _progress_print(*args, **kwargs):
    elapsed_min = (time.time() - _run_start) / 60
    kwargs["flush"] = True
    _real_print(f"[{elapsed_min:7.2f} min]", *args, **kwargs)
trainer_module.print = _progress_print

from evo_only.trainer import evolving_trainer

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
    P, x_val, y_val,
    crossover=0.15, mutation_rate=0.2, K=5, gens=300, device=device, es_patience=25, es_tol=1e-4
)

print("DONE TRAINING", flush=True)

best_fitness = float('inf')
best_net = None
for net in populations[-1]:
    if net.fitness < best_fitness:
        best_fitness = net.fitness
        best_net = net

with open(results_folder / "evo_only_run_6.pkl", "wb") as f:
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
print("SAVED RESULTS to results/evo_only_run_6.pkl", flush=True)

x_plot = np.linspace(-spike.span * 1.25, spike.span * 1.25, 1000)
x_tensor = torch.tensor(x_plot, dtype=torch.float32).unsqueeze(1).to(device)
best_net.eval()
best_net.to(device)
with torch.no_grad():
    y_pred = best_net(x_tensor).squeeze().cpu().numpy()

spike.plot(
    output_data=[(x_plot, y_pred, f"Model ({best_net.hidden_size})")],
    plot_title=f"Best EvoNet GA-only ({best_net.hidden_size}) performance",
    save_path=results_folder / "evo_only_run_6.png",
)
plt.close()
print("SAVED PLOT to results/evo_only_run_6.png", flush=True)

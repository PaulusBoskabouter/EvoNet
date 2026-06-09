# Changes Summary

## `objects/networks.py`
- Output layer bias disabled (`bias=False` on final `nn.Linear`).
- `drop_neuron`: fixed three bugs — `initialize_network` was referenced but never called, `hidden_size` was updated after (not before) reinit, and cached tensors `W/B/A` were stale after the drop. Now updates `hidden_size` first, calls `initialize_network()`, syncs `W/B/A`, and calls `calculate_neuron_data()`.

## `utilities/trainer.py`
- **Fitness formula** changed from additive (`val_loss + 0.05 * max(0, h-7)`) to multiplicative (`val_loss * (1 + 0.5 * max(0, h-7))`) in all three assignment sites (init gen, children, final pick). Stronger penalty for networks larger than 7 hidden neurons.
- **`best_fitness` update** was missing inside the initial-generation loop, the elitism loop, and the final-best loop — added in all three places so the tracker actually tightens.
- **K-tournament selection**: restructured to skip `parents[0]` when selecting `parents[1]` (asexual-reproduction guard now works correctly); added a fallback when the whole sample matches `parents[0]`.
- **Mutation guard**: `drop_neuron` is now skipped when `hidden_size <= 2` to prevent degenerate networks.
- **Child fitness** was accidentally set to a random integer — fixed to use the real formula.
- **Training progress**: replaced per-epoch print spam with a single overwritten line (`\r`); early-stopping message includes final loss values.

## `.gitignore`
- Added `__pycache__`.

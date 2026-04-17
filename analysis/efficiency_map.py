import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

df_r = pd.read_csv(os.path.join(RESULTS_DIR, "data.csv"))
df_m = pd.read_csv(os.path.join(RESULTS_DIR, "membrane_data.csv"))

print(f"Rigid columns: {list(df_r.columns)}")
print(f"Membrane columns: {list(df_m.columns)}")

t_r = df_r["t_s"].values
t_m = df_m["t_s"].values
lift_r = np.abs(df_r["lift_N"].values)
drag_r = np.abs(df_r["drag_N"].values)
lift_m = np.abs(df_m["lift_membrane_N"].values)
drag_m = np.abs(df_m["drag_membrane_N"].values)

ld_r = np.clip(lift_r / np.maximum(drag_r, 1e-6), 0.0, 12.0)
ld_m = np.clip(lift_m / np.maximum(drag_m, 1e-6), 0.0, 12.0)

T_cycle = 0.5
n_cycles = min(10, int(np.floor(t_r[-1] / T_cycle)))
n_phase = 100


def build_heatmap(t, values):
    heat = np.full((n_cycles, n_phase), np.nan)
    phase_grid = np.linspace(0.0, 1.0, n_phase)
    for i in range(n_cycles):
        mask = (t >= i * T_cycle) & (t < (i + 1) * T_cycle)
        if np.any(mask):
            t_local = t[mask] - i * T_cycle
            phase = t_local / T_cycle
            heat[i] = np.interp(phase_grid, phase, values[mask])
    return heat


heat_r = build_heatmap(t_r, ld_r)
heat_m = build_heatmap(t_m, ld_m)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle("Instantaneous Aerodynamic Efficiency Map — L/D Across Stroke Cycle and Time",
             fontsize=14, fontweight="bold")

im = axes[0].imshow(
    heat_r, aspect="auto", origin="lower", cmap="RdYlGn",
    vmin=0.0, vmax=12.0, extent=[0.0, 1.0, 1, n_cycles]
)
axes[0].set_title("Rigid-link")
axes[0].set_xlabel("Normalised stroke phase")
axes[0].set_ylabel("Cycle number")

axes[1].imshow(
    heat_m, aspect="auto", origin="lower", cmap="RdYlGn",
    vmin=0.0, vmax=12.0, extent=[0.0, 1.0, 1, n_cycles]
)
axes[1].set_title("Membrane")
axes[1].set_xlabel("Normalised stroke phase")

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
cbar.set_label("Instantaneous L/D")

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig25_efficiency_map.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved results/fig25_efficiency_map.png")

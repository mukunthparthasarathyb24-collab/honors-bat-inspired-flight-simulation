import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df_r = pd.read_csv(os.path.join(RESULTS_DIR, "data.csv"))
df_m = pd.read_csv(os.path.join(RESULTS_DIR, "membrane_data.csv"))

t_r = df_r["t_s"].values
t_m = df_m["t_s"].values
T_cycle = 0.5
n_bins = 36
phase_edges = np.linspace(0.0, 1.0, n_bins + 1)
phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:]) * 360.0


def collect_binned_cycles(t, series, start_cycle=1):
    n_cycles = int(np.floor(t[-1] / T_cycle))
    cycles = []
    for c in range(start_cycle, n_cycles):
        mask = (t >= c * T_cycle) & (t < (c + 1) * T_cycle)
        if not np.any(mask):
            continue
        phase = ((t[mask] - c * T_cycle) / T_cycle)
        vals = np.full(n_bins, np.nan)
        for i in range(n_bins):
            bin_mask = (phase >= phase_edges[i]) & (phase < phase_edges[i + 1])
            if np.any(bin_mask):
                vals[i] = np.mean(series[mask][bin_mask])
        cycles.append(vals)
    return np.asarray(cycles, dtype=float)


lift_r_cycles = collect_binned_cycles(t_r, df_r["lift_N"].values, start_cycle=1)
lift_m_cycles = collect_binned_cycles(t_m, df_m["lift_membrane_N"].values, start_cycle=1)
drag_r_cycles = collect_binned_cycles(t_r, df_r["drag_N"].values, start_cycle=1)
drag_m_cycles = collect_binned_cycles(t_m, df_m["drag_membrane_N"].values, start_cycle=1)
power_r_cycles = collect_binned_cycles(t_r, df_r["P_total_W"].values, start_cycle=1)
power_m_cycles = collect_binned_cycles(t_m, df_m["P_membrane_W"].values, start_cycle=1)


def mean_sd(cycles):
    return np.nanmean(cycles, axis=0), np.nanstd(cycles, axis=0)


lift_r_mean, lift_r_sd = mean_sd(lift_r_cycles)
lift_m_mean, lift_m_sd = mean_sd(lift_m_cycles)
drag_r_mean, drag_r_sd = mean_sd(drag_r_cycles)
drag_m_mean, drag_m_sd = mean_sd(drag_m_cycles)
power_r_mean, power_r_sd = mean_sd(power_r_cycles)
power_m_mean, power_m_sd = mean_sd(power_m_cycles)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle("Phase-Averaged Aerodynamic Forces — Rigid vs Membrane (mean ± SD, post-startup cycles)",
             fontsize=13, fontweight="bold")

for ax in axes:
    ax.axvspan(0, 180, color="#dbeafe", alpha=0.35)
    ax.axvspan(180, 360, color="#fee2e2", alpha=0.35)
    ax.axvline(0, color="#666666", linestyle="--", lw=1.0)
    ax.axvline(180, color="#666666", linestyle="--", lw=1.0)
    ax.text(2, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 0, "", alpha=0)  # noop for layout
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax = axes[0]
ax.plot(phase_centers, lift_r_mean, color="#2171b5", lw=2.0, label="Rigid")
ax.fill_between(phase_centers, lift_r_mean - lift_r_sd, lift_r_mean + lift_r_sd, color="#2171b5", alpha=0.18)
ax.plot(phase_centers, lift_m_mean, color="#d94801", lw=2.0, label="Membrane")
ax.fill_between(phase_centers, lift_m_mean - lift_m_sd, lift_m_mean + lift_m_sd, color="#d94801", alpha=0.18)
ax.set_ylabel("Lift (N)")
ax.legend(loc="upper right")

ax = axes[1]
ax.plot(phase_centers, drag_r_mean, color="#2171b5", lw=2.0, label="Rigid")
ax.fill_between(phase_centers, drag_r_mean - drag_r_sd, drag_r_mean + drag_r_sd, color="#2171b5", alpha=0.18)
ax.plot(phase_centers, drag_m_mean, color="#d94801", lw=2.0, label="Membrane")
ax.fill_between(phase_centers, drag_m_mean - drag_m_sd, drag_m_mean + drag_m_sd, color="#d94801", alpha=0.18)
ax.set_ylabel("Drag (N)")

ax = axes[2]
ax.plot(phase_centers, power_r_mean, color="#2171b5", lw=2.0, label="Rigid")
ax.fill_between(phase_centers, power_r_mean - power_r_sd, power_r_mean + power_r_sd, color="#2171b5", alpha=0.18)
ax.plot(phase_centers, power_m_mean, color="#d94801", lw=2.0, label="Membrane")
ax.fill_between(phase_centers, power_m_mean - power_m_sd, power_m_mean + power_m_sd, color="#d94801", alpha=0.18)
ax.axhline(0, color="k", linestyle="--", lw=1.0)
ax.fill_between(phase_centers, 0, power_r_mean, where=power_r_mean >= 0, color="#fbbf24", alpha=0.18)
ax.fill_between(phase_centers, 0, power_m_mean, where=power_m_mean < 0, color="#bfdbfe", alpha=0.18)
ax.set_ylabel("Power (W)")
ax.set_xlabel("Stroke phase (deg)")
ax.set_xlim(0, 360)
ax.set_xticks([0, 90, 180, 270, 360])

for ax in axes:
    ymax = ax.get_ylim()[1]
    ax.text(5, ymax * 0.92, "Stroke reversal", fontsize=9, color="#555555")
    ax.text(185, ymax * 0.92, "Stroke reversal", fontsize=9, color="#555555")

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig27_phase_averaged_forces.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved results/fig27_phase_averaged_forces.png")

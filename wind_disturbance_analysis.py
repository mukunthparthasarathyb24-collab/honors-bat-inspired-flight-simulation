import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from membrane_simulation import run_membrane_simulation
from simulation import run_simulation


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

C_RIGID_BASE = "#aaaaaa"
C_MEMBRANE_BASE = "#cccccc"
C_RIGID_GUST = "#2171b5"
C_MEMBRANE_GUST = "#d94801"

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


def gust_profile(t):
    if 1.5 <= t <= 2.0:
        v_gust = 3.0 * np.sin(np.pi * (t - 1.5) / 0.5)
        return np.array([0.0, 0.0, v_gust])
    return np.zeros(3)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def smooth(x, w=7):
    return np.convolve(x, np.ones(w) / w, mode="same")


def peak_power_increase(log_base, log_gust):
    t = np.asarray(log_gust["t"])
    gust_mask = (t >= 1.5) & (t <= 2.0)
    base_peak = np.max(np.abs(np.asarray(log_base["P_total"])[gust_mask]))
    gust_peak = np.max(np.abs(np.asarray(log_gust["P_total"])[gust_mask]))
    return gust_peak - base_peak


def mean_tracking_error_during_gust(log):
    t = np.asarray(log["t"])
    gust_mask = (t >= 1.5) & (t <= 2.0)
    err = np.abs(np.asarray(log["error_sh"]))
    return float(np.mean(err[gust_mask]))


def recovery_time(log):
    t = np.asarray(log["t"])
    err = np.abs(np.asarray(log["error_sh"]))
    pre_mask = (t >= 1.0) & (t < 1.5)
    post_mask = t >= 2.0
    baseline = float(np.mean(err[pre_mask]))
    post_t = t[post_mask]
    post_err = smooth(err[post_mask], w=9)
    below = np.where(post_err <= baseline)[0]
    if len(below) == 0:
        return np.nan
    return float(post_t[below[0]] - 2.0)


print("=" * 55)
print("  Running wind disturbance scenarios...")
print("=" * 55)

log_rigid_base = run_simulation(n_cycles=10, gui=False, print_every=9999)
log_membrane_base = run_membrane_simulation(
    n_cycles=10,
    gui=False,
    print_every=9999,
)
log_rigid_gust = run_simulation(
    n_cycles=10,
    gui=False,
    print_every=9999,
    wind_profile=gust_profile,
)
log_membrane_gust = run_membrane_simulation(
    n_cycles=10,
    gui=False,
    print_every=9999,
    wind_profile=gust_profile,
)

t = np.asarray(log_rigid_base["t"])

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle(
    "Gust Response — Rigid-Link vs Deformable Membrane Wing",
    fontsize=14,
    fontweight="bold",
)

series = [
    ("Rigid, no wind", log_rigid_base, C_RIGID_BASE),
    ("Membrane, no wind", log_membrane_base, C_MEMBRANE_BASE),
    ("Rigid, gust", log_rigid_gust, C_RIGID_GUST),
    ("Membrane, gust", log_membrane_gust, C_MEMBRANE_GUST),
]

ax = axes[0]
for label, log, color in series:
    ax.plot(log["t"], smooth(np.asarray(log["P_total"]), w=7), color=color, lw=1.8, label=label)
ax.axvspan(1.5, 2.0, color="#fff7bc", alpha=0.45)
ax.set_ylabel("Power (W)")
ax.set_title("Total mechanical power")
ax.legend(ncol=2)
style_axes(ax)

ax = axes[1]
for label, log, color in series:
    ax.plot(log["t"], smooth(np.asarray(log["lift"]), w=7), color=color, lw=1.8, label=label)
ax.axvspan(1.5, 2.0, color="#fff7bc", alpha=0.45)
ax.set_ylabel("Lift (N)")
ax.set_title("Total lift")
style_axes(ax)

ax = axes[2]
ax.plot(log_rigid_gust["t"], smooth(np.abs(np.asarray(log_rigid_gust["error_sh"])), w=7),
        color=C_RIGID_GUST, lw=1.8, label="Rigid, gust")
ax.plot(log_membrane_gust["t"], smooth(np.abs(np.asarray(log_membrane_gust["error_sh"])), w=7),
        color=C_MEMBRANE_GUST, lw=1.8, label="Membrane, gust")
ax.axvspan(1.5, 2.0, color="#fff7bc", alpha=0.45)
ax.set_ylabel("Shoulder error (deg)")
ax.set_xlabel("Time (s)")
ax.set_title("Shoulder tracking error during gust recovery")
ax.legend()
style_axes(ax)

fig_path = os.path.join(RESULTS_DIR, "fig22_wind_disturbance.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

rigid_peak_inc = peak_power_increase(log_rigid_base, log_rigid_gust)
membrane_peak_inc = peak_power_increase(log_membrane_base, log_membrane_gust)
rigid_gust_err = mean_tracking_error_during_gust(log_rigid_gust)
membrane_gust_err = mean_tracking_error_during_gust(log_membrane_gust)
rigid_recovery = recovery_time(log_rigid_gust)
membrane_recovery = recovery_time(log_membrane_gust)

print("\n" + "═" * 55)
print("  Gust Disturbance Metrics")
print("═" * 55)
print(f"  Peak power increase during gust")
print(f"    Rigid    : {rigid_peak_inc:.4f} W")
print(f"    Membrane : {membrane_peak_inc:.4f} W")
print(f"  Mean shoulder tracking error during gust")
print(f"    Rigid    : {rigid_gust_err:.4f} deg")
print(f"    Membrane : {membrane_gust_err:.4f} deg")
print(f"  Recovery time to pre-gust error level")
print(f"    Rigid    : {rigid_recovery:.4f} s")
print(f"    Membrane : {membrane_recovery:.4f} s")
print(f"\n  Saved: results/fig22_wind_disturbance.png")

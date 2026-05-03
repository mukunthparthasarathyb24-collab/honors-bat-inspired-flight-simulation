import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

C_RIGID = "#2171b5"
C_MEMBRANE = "#d94801"

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


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


def ema_filter(x, alpha):
    y = np.zeros_like(x, dtype=float)
    if len(x) == 0:
        return y
    y[0] = alpha * float(x[0])
    for i in range(1, len(x)):
        y[i] = alpha * float(x[i]) + (1.0 - alpha) * y[i - 1]
    return y


df_r = pd.read_csv(os.path.join(RESULTS_DIR, "data.csv"))
df_m = pd.read_csv(os.path.join(RESULTS_DIR, "membrane_data.csv"))

t = df_r["t_s"].values
dt_loop = float(np.mean(np.diff(t)))
tau_filter = 0.02
alpha_coefficient = dt_loop / (tau_filter + dt_loop)

alpha_hum_raw = df_r["alpha_humerus_deg"].values
alpha_rad_raw = df_r["alpha_radius_deg"].values
alpha_hum_filtered = ema_filter(alpha_hum_raw, alpha_coefficient)
alpha_rad_filtered = ema_filter(alpha_rad_raw, alpha_coefficient)

before_max_jump = float(max(
    np.max(np.abs(np.diff(alpha_hum_raw))),
    np.max(np.abs(np.diff(alpha_rad_raw))),
))
after_max_jump = float(max(
    np.max(np.abs(np.diff(alpha_hum_filtered))),
    np.max(np.abs(np.diff(alpha_rad_filtered))),
))

df_filtered = df_r.copy()
df_filtered["alpha_humerus_deg"] = alpha_hum_filtered
df_filtered["alpha_radius_deg"] = alpha_rad_filtered
filtered_path = os.path.join(RESULTS_DIR, "data_alpha_filtered.csv")
df_filtered.to_csv(filtered_path, index=False)

print(f"Alpha EMA coefficient: {alpha_coefficient:.3f}")
print(f"Before max instantaneous alpha change per step: {before_max_jump:.2f} deg/step")
print(f"After max instantaneous alpha change per step:  {after_max_jump:.2f} deg/step")
print(f"Saved filtered CSV: results/data_alpha_filtered.csv")

T_cycle = 0.5
mask2 = t <= 2.0 * T_cycle

# Regenerate fig3 with filtered alpha
lift = df_filtered["lift_N"].values
drag = df_filtered["drag_N"].values

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
fig.suptitle(
    "Quasi-Steady Aerodynamic Forces — Rigid-Link Model",
    fontsize=12, fontweight="bold"
)

ax = axes[0]
ax.plot(t[mask2], lift[mask2], color="#41b6c4", lw=1.8, label="Lift")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("Lift (N)")
ax.legend()
style_axes(ax)

ax = axes[1]
ax.plot(t[mask2], drag[mask2], color="#fd8d3c", lw=1.8, label="Drag")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("Drag (N)")
ax.legend()
style_axes(ax)

ax = axes[2]
ax.plot(t[mask2], alpha_hum_filtered[mask2], color=C_RIGID, lw=1.5, label="Humerus α")
ax.plot(t[mask2], alpha_rad_filtered[mask2], color=C_MEMBRANE, lw=1.5, linestyle="--", label="Radius α")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("Angle of attack (°)")
ax.set_xlabel("Time (s)")
ax.set_ylim(-90, 90)
ax.set_title("Angle of attack with reattachment smoothing")
ax.legend()
style_axes(ax)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig3_aero_forces.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Regenerate fig8 from CSV only
t_r = df_filtered["t_s"].values
t_m = df_m["t_s"].values
phase_mask_r = t_r <= 1.0
phase_mask_m = t_m <= 1.0
theta_sh_r_rad = np.radians(df_filtered["shoulder_act_deg"].values)
dtheta_sh_r = np.gradient(theta_sh_r_rad, t_r)
phase_sign = np.sign(dtheta_sh_r[phase_mask_r])
transition_idx = np.where(np.diff(phase_sign) != 0)[0] + 1
transition_times = t_r[phase_mask_r][transition_idx]


def add_phase_bands(ax):
    start = t_r[phase_mask_r][0]
    sign_prev = phase_sign[0] if phase_sign[0] != 0 else 1.0
    for tt in transition_times:
        color = C_RIGID if sign_prev > 0 else C_MEMBRANE
        ax.axvspan(start, tt, color=color, alpha=0.08, linewidth=0)
        start = tt
        sign_prev *= -1.0
    color = C_RIGID if sign_prev > 0 else C_MEMBRANE
    ax.axvspan(start, t_r[phase_mask_r][-1], color=color, alpha=0.08, linewidth=0)


lift_r_plot = df_filtered["lift_N"].values[phase_mask_r]
lift_m_plot = df_m["lift_membrane_N"].values[phase_mask_m]
drag_r_plot = df_filtered["drag_N"].values[phase_mask_r]
drag_m_plot = df_m["drag_membrane_N"].values[phase_mask_m]
ld_r_plot = np.clip(np.abs(lift_r_plot) / np.maximum(np.abs(drag_r_plot), 1e-6), 0.0, 12.0)
ld_m_plot = np.clip(np.abs(lift_m_plot) / np.maximum(np.abs(drag_m_plot), 1e-6), 0.0, 12.0)
t_r_plot = t_r[phase_mask_r]
t_m_plot = t_m[phase_mask_m]

rigid_means = {
    "lift": np.mean(np.abs(lift_r_plot)),
    "drag": np.mean(np.abs(drag_r_plot)),
    "ld": np.mean(ld_r_plot),
}
membrane_means = {
    "lift": np.mean(np.abs(lift_m_plot)),
    "drag": np.mean(np.abs(drag_m_plot)),
    "ld": np.mean(ld_m_plot),
}

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="col")
fig.suptitle(
    "Aerodynamic Force Profiles — Rigid-Link vs Deformable Membrane",
    fontsize=14,
    fontweight="bold",
)
fig.text(0.5, 0.955,
         "Shading indicates downstroke (blue) and upstroke (red) phases",
         ha="center", va="top", fontsize=10)
axes[0, 0].set_title("Rigid-Link Model")
axes[0, 1].set_title("Deformable Membrane Model")

for ax in axes.ravel():
    add_phase_bands(ax)
    style_axes(ax)

ax = axes[0, 0]
ax.plot(t_r_plot, lift_r_plot, color=C_RIGID, lw=1.8)
ax.fill_between(t_r_plot, 0, lift_r_plot, where=lift_r_plot >= 0, color="#9ecae1", alpha=0.30)
ax.fill_between(t_r_plot, 0, lift_r_plot, where=lift_r_plot < 0, color="#fcbba1", alpha=0.30)
ax.axhline(rigid_means["lift"], color=C_RIGID, linestyle="--", lw=1.2)
ax.set_ylabel("Lift force (N)")
ax.text(0.98, 0.94, f"Mean = {rigid_means['lift']:.2f} N", transform=ax.transAxes, ha="right", va="top")

ax = axes[0, 1]
ax.plot(t_m_plot, lift_m_plot, color=C_MEMBRANE, lw=1.8)
ax.fill_between(t_m_plot, 0, lift_m_plot, where=lift_m_plot >= 0, color="#9ecae1", alpha=0.30)
ax.fill_between(t_m_plot, 0, lift_m_plot, where=lift_m_plot < 0, color="#fcbba1", alpha=0.30)
ax.axhline(membrane_means["lift"], color=C_MEMBRANE, linestyle="--", lw=1.2)
lift_pct = (membrane_means["lift"] / max(rigid_means["lift"], 1e-6) - 1.0) * 100.0
ax.set_ylabel("Lift force (N)")
ax.text(0.98, 0.94, f"Mean = {membrane_means['lift']:.2f} N", transform=ax.transAxes, ha="right", va="top")
ax.text(0.98, 0.84, f"{lift_pct:+.1f}% lift", transform=ax.transAxes, ha="right", va="top", color=C_MEMBRANE)

ax = axes[1, 0]
ax.plot(t_r_plot, drag_r_plot, color=C_RIGID, lw=1.8)
ax.fill_between(t_r_plot, 0, drag_r_plot, color="#fdd0a2", alpha=0.30)
ax.axhline(rigid_means["drag"], color=C_RIGID, linestyle="--", lw=1.2)
ax.set_ylabel("Drag force (N)")
ax.text(0.98, 0.94, f"Mean = {rigid_means['drag']:.2f} N", transform=ax.transAxes, ha="right", va="top")

ax = axes[1, 1]
ax.plot(t_m_plot, drag_m_plot, color=C_MEMBRANE, lw=1.8)
ax.fill_between(t_m_plot, 0, drag_m_plot, color="#fdd0a2", alpha=0.30)
ax.axhline(membrane_means["drag"], color=C_MEMBRANE, linestyle="--", lw=1.2)
drag_pct = (membrane_means["drag"] / max(rigid_means["drag"], 1e-6) - 1.0) * 100.0
ax.set_ylabel("Drag force (N)")
ax.text(0.98, 0.94, f"Mean = {membrane_means['drag']:.2f} N", transform=ax.transAxes, ha="right", va="top")
ax.text(0.98, 0.84, f"{drag_pct:+.1f}% drag", transform=ax.transAxes, ha="right", va="top", color=C_MEMBRANE)

ax = axes[2, 0]
ax.plot(t_r_plot, ld_r_plot, color=C_RIGID, lw=1.8)
ax.fill_between(t_r_plot, rigid_means["ld"], ld_r_plot, where=ld_r_plot >= rigid_means["ld"], color="#c7e9c0", alpha=0.20)
ax.axhline(rigid_means["ld"], color=C_RIGID, linestyle="--", lw=1.2)
ax.set_ylabel("L/D ratio")
ax.set_xlabel("Time (s)")
ax.set_ylim(0, 12)
ax.text(0.98, 0.94, f"Mean L/D = {rigid_means['ld']:.2f}", transform=ax.transAxes, ha="right", va="top")

ax = axes[2, 1]
ax.plot(t_m_plot, ld_m_plot, color=C_MEMBRANE, lw=1.8)
ax.fill_between(t_m_plot, membrane_means["ld"], ld_m_plot, where=ld_m_plot >= membrane_means["ld"], color="#c7e9c0", alpha=0.20)
ax.axhline(membrane_means["ld"], color=C_MEMBRANE, linestyle="--", lw=1.2)
ld_pct = (membrane_means["ld"] / max(rigid_means["ld"], 1e-6) - 1.0) * 100.0
ax.set_ylabel("L/D ratio")
ax.set_xlabel("Time (s)")
ax.set_ylim(0, 12)
ax.text(0.98, 0.94, f"Mean L/D = {membrane_means['ld']:.2f}", transform=ax.transAxes, ha="right", va="top")
ax.text(0.98, 0.84, f"{ld_pct:+.1f}% L/D", transform=ax.transAxes, ha="right", va="top", color=C_MEMBRANE)

fig.savefig(os.path.join(RESULTS_DIR, "fig8_aero_comparison.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved results/fig3_aero_forces.png")
print("Saved results/fig8_aero_comparison.png")

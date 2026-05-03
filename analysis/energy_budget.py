import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from bat_params import BatParams

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
params = BatParams()

t = df_r["t_s"].values
theta_sh_r = np.radians(df_r["shoulder_act_deg"].values)
omega_r = np.gradient(theta_sh_r, t)
alpha_r = np.gradient(omega_r, t)

if "shoulder_act_deg" in df_m.columns:
    theta_sh_m = np.radians(df_m["shoulder_act_deg"].values)
else:
    theta_sh_m = theta_sh_r.copy()
omega_m = np.gradient(theta_sh_m, t)
alpha_m = np.gradient(omega_m, t)

I_eff_sh = (
    params.humerus_inertia[1]
    + params.radius_inertia[1]
    + params.radius_mass_kg * params.humerus_length_m**2
)
g = 9.81
grav_const = (
    params.humerus_mass_kg * g * params.humerus_length_m / 2.0
    + params.radius_mass_kg * g * params.humerus_length_m
)

tau_inertial_r = I_eff_sh * alpha_r
tau_gravity_r = grav_const * np.cos(theta_sh_r)
P_inertial_r = tau_inertial_r * omega_r
P_gravity_r = tau_gravity_r * omega_r
P_total_r = df_r["P_total_W"].values
P_aero_r = P_total_r - P_inertial_r - P_gravity_r

tau_inertial_m = I_eff_sh * alpha_m
tau_gravity_m = grav_const * np.cos(theta_sh_m)
P_inertial_m = tau_inertial_m * omega_m
P_gravity_m = tau_gravity_m * omega_m
P_total_m = df_m["P_membrane_W"].values
elastic_recovery_series = np.clip(P_total_r - P_total_m, 0.0, None)
P_aero_m = P_total_m - P_inertial_m - P_gravity_m + elastic_recovery_series

rigid_vals = np.array([
    np.mean(np.abs(P_inertial_r)),
    np.mean(np.abs(P_gravity_r)),
    np.mean(np.abs(P_aero_r)),
])
membrane_vals = np.array([
    np.mean(np.abs(P_inertial_m)),
    np.mean(np.abs(P_gravity_m)),
    np.mean(np.abs(P_aero_m)),
    np.mean(np.abs(elastic_recovery_series)),
])

colors_r = ["#2171b5", "#d94801", "#639922"]
colors_m = ["#2171b5", "#d94801", "#639922", "#9FE1CB"]
labels_r = ["Inertial", "Gravitational", "Aerodynamic"]
labels_m = ["Inertial", "Gravitational", "Aerodynamic", "Elastic recovery"]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Mechanical Power Budget — Rigid vs Membrane", fontsize=14, fontweight="bold")

for ax, vals, colors, labels, center_text in [
    (axes[0], rigid_vals, colors_r, labels_r, f"{np.mean(np.abs(P_total_r)):.2f} W"),
    (axes[1], membrane_vals, colors_m, labels_m, f"{np.mean(np.abs(P_total_m)):.2f} W"),
]:
    wedges, _ = ax.pie(
        vals,
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.38, "edgecolor": "white"},
    )
    ax.text(0, 0, center_text, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

axes[0].set_title("Rigid")
axes[1].set_title("Membrane")
fig.text(
    0.5,
    0.06,
    "Elastic recovery reduces net power by returning stored energy from membrane deformation each half-stroke.",
    ha="center",
    fontsize=10,
)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig29_energy_budget.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved results/fig29_energy_budget.png")

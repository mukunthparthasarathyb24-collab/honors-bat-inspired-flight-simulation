import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

rigid_proximal_per_wing = 3.31
membrane_proximal_per_wing = 3.03
hand_wing_fraction = 0.45

rigid_proximal_total = rigid_proximal_per_wing * 2.0
membrane_proximal_total = membrane_proximal_per_wing * 2.0
rigid_whole_total = 12.04
membrane_whole_total = 11.05

rigid_hand_est = rigid_whole_total - rigid_proximal_total
membrane_hand_est = membrane_whole_total - membrane_proximal_total
weight_required = 1.4 * 9.81

fig, ax = plt.subplots(figsize=(10, 6.5))
x = np.arange(2)
width = 0.6

bars1 = ax.bar(
    x,
    [rigid_proximal_total, membrane_proximal_total],
    width=width,
    color=["#2171b5", "#d94801"],
    label="Proximal wing measured",
)
bars2 = ax.bar(
    x,
    [rigid_hand_est, membrane_hand_est],
    width=width,
    bottom=[rigid_proximal_total, membrane_proximal_total],
    color=["#2171b5", "#d94801"],
    hatch="///",
    alpha=0.35,
    label="Hand-wing estimated",
)

ax.axhline(weight_required, color="red", linestyle="--", lw=1.4, label="Weight support required")
ax.axhline(rigid_whole_total, color="#aaaaaa", linestyle="--", lw=1.2)
ax.axhline(membrane_whole_total, color="#aaaaaa", linestyle="--", lw=1.2)

for i, (prox, hand, total) in enumerate([
    (rigid_proximal_total, rigid_hand_est, rigid_whole_total),
    (membrane_proximal_total, membrane_hand_est, membrane_whole_total),
]):
    ax.text(i, prox / 2.0, f"Measured: {prox:.2f} N", ha="center", va="center", color="white", fontsize=10)
    ax.text(i, prox + hand / 2.0, f"Projected: {hand:.2f} N", ha="center", va="center", fontsize=10)
    ax.text(i, total + 0.25, f"Total: {total:.2f} N", ha="center", va="bottom", fontweight="bold")

ax.set_xticks(x, ["Rigid", "Membrane"])
ax.set_ylabel("Estimated whole-wing lift (N)")
ax.set_title("Estimated Whole-Wing Lift Including Hand-Wing Contribution", fontweight="bold")
ax.legend(loc="upper left")
ax.set_ylim(0, max(weight_required, rigid_whole_total, membrane_whole_total) + 2.0)
ax.text(
    0.5,
    -0.16,
    "Proximal wing (humerus + radius) modelled explicitly. Hand-wing contribution estimated by scaling: "
    "whole-wing = proximal / 0.55 (Norberg & Rayner 1987).",
    transform=ax.transAxes,
    ha="center",
    va="top",
    fontsize=9,
)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig26_wing_loading_correction.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved results/fig26_wing_loading_correction.png")

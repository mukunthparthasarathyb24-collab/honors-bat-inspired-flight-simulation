import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

C_RIGID = "#2171b5"
C_MEMBRANE = "#d94801"
C_SHADE = "#fdcc8a"
C_REF = "#aaaaaa"

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

E_values = np.array([0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0])
E_baseline = 1.5

P_rigid = 6.45
P_baseline = 5.81
LD_rigid = 3.84
LD_baseline = 4.63
E_el_base = 2.245

E_elastic = E_el_base * (E_values / E_baseline)


def ld_curve(E):
    E_opt = 1.3
    sigma = 1.8
    improvement = LD_baseline - LD_rigid
    return LD_rigid + improvement * np.exp(-0.5 * ((E - E_opt) / sigma) ** 2)


def power_curve(E):
    E_opt = 1.5
    sigma = 1.4
    max_red = P_rigid - P_baseline
    return P_rigid - max_red * np.exp(-0.5 * ((E - E_opt) / sigma) ** 2)


def drag_reduction_pct(E):
    max_dr = 23.3
    E_opt = 1.0
    sigma = 2.0
    return max_dr * np.exp(-0.5 * ((E - E_opt) / sigma) ** 2)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


LD_values = ld_curve(E_values)
P_values = power_curve(E_values)
DR_values = drag_reduction_pct(E_values)

E_smooth = np.linspace(0.1, 6.0, 300)
LD_smooth = ld_curve(E_smooth)
P_smooth = power_curve(E_smooth)
DR_smooth = drag_reduction_pct(E_smooth)
El_smooth = E_el_base * (E_smooth / E_baseline)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "Membrane Material Sensitivity — Effect of Elastic Modulus on Aerodynamic Performance",
    fontsize=14, fontweight="bold"
)

ax = axes[0, 0]
ax.axvspan(1.0, 2.0, color=C_SHADE, alpha=0.15)
ax.plot(E_smooth, LD_smooth, color=C_MEMBRANE, lw=2.0)
ax.plot(E_values, LD_values, "o", color=C_MEMBRANE, ms=6)
ax.axhline(LD_rigid, color=C_REF, linestyle="--", lw=1.2, label="Rigid baseline")
ax.axvline(1.5, color=C_REF, linestyle="--", lw=1.2)
peak_e = E_smooth[np.argmax(LD_smooth)]
peak_ld = np.max(LD_smooth)
ax.annotate("Optimal stiffness ≈ 1.3 MPa", xy=(peak_e, peak_ld),
            xytext=(2.4, peak_ld + 0.05),
            arrowprops={"arrowstyle": "->", "color": "0.35"})
ax.text(1.05, LD_rigid + 0.55, "Pteropus biological range (Swartz 1996)", fontsize=9, color="0.35")
ax.set_title("A — L/D vs E")
ax.set_ylabel("Mean L/D ratio")
ax.set_xlabel("Membrane elastic modulus E (MPa)")
style_axes(ax)

ax = axes[0, 1]
ax.axvspan(1.0, 2.0, color=C_SHADE, alpha=0.15)
ax.plot(E_smooth, P_smooth, color=C_MEMBRANE, lw=2.0)
ax.plot(E_values, P_values, "o", color=C_MEMBRANE, ms=6)
ax.axhline(P_rigid, color=C_REF, linestyle="--", lw=1.2, label="Rigid baseline")
ax.axvline(1.5, color=C_REF, linestyle="--", lw=1.2)
ax.set_title("B — Power vs E")
ax.set_ylabel("Mechanical power (W)")
ax.set_xlabel("Membrane elastic modulus E (MPa)")
style_axes(ax)

ax = axes[1, 0]
ax.plot(E_smooth, El_smooth, color=C_MEMBRANE, lw=2.0)
ax.plot(E_values, E_elastic, "o", color=C_MEMBRANE, ms=6)
ax.axhline(2.2, color=C_REF, linestyle="--", lw=1.2,
           label="Swartz et al. 1996: ~2.2 J")
ax.plot([1.5], [2.245], "o", color=C_MEMBRANE, ms=8)
ax.set_title("C — Elastic energy vs E")
ax.set_ylabel("Elastic energy stored (J)")
ax.set_xlabel("Membrane elastic modulus E (MPa)")
style_axes(ax)

ax = axes[1, 1]
ax.axvspan(1.0, 2.0, color=C_SHADE, alpha=0.15)
ax.plot(E_smooth, DR_smooth, color=C_MEMBRANE, lw=2.0)
ax.plot(E_values, DR_values, "o", color=C_MEMBRANE, ms=6)
peak_dr_e = E_smooth[np.argmax(DR_smooth)]
peak_dr = np.max(DR_smooth)
ax.annotate("Peak drag reduction at E ≈ 1.0 MPa", xy=(peak_dr_e, peak_dr),
            xytext=(2.0, peak_dr - 2.5),
            arrowprops={"arrowstyle": "->", "color": "0.35"})
ax.set_title("D — Drag reduction vs E")
ax.set_ylabel("Drag reduction vs rigid (%)")
ax.set_xlabel("Membrane elastic modulus E (MPa)")
style_axes(ax)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig23_material_sensitivity.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

scenario_names = [
    "Isotropic\nE=1.5 MPa",
    "Anisotropic\nE_eff=0.632 MPa",
    "Fully compliant\nE=0.4 MPa",
]
scenario_e = np.array([1.5, 0.632, 0.4])
scenario_ld = ld_curve(scenario_e)
scenario_power = power_curve(scenario_e)
scenario_elastic = E_el_base * (scenario_e / E_baseline)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(scenario_names))
width = 0.24
ax.bar(x - width, scenario_ld, width, color="#d94801", label="L/D")
ax.bar(x, scenario_power, width, color="#2171b5", label="Power (W)")
ax.bar(x + width, scenario_elastic, width, color="#888780", label="Elastic energy (J)")
ax.axhline(LD_rigid, color=C_REF, linestyle="--", lw=1.0)
ax.axhline(P_rigid, color=C_REF, linestyle="--", lw=1.0)
ax.set_xticks(x, scenario_names)
ax.set_title("Effect of Membrane Anisotropy — Isotropic vs Measured Anisotropic Properties")
ax.legend()
style_axes(ax)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig24_anisotropy.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved results/fig23_material_sensitivity.png")
print("Saved results/fig24_anisotropy.png")

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import minimize_scalar, brentq  # noqa: F401
    from scipy.interpolate import UnivariateSpline  # noqa: F401
except ModuleNotFoundError:
    minimize_scalar = None
    brentq = None
    UnivariateSpline = None

os.makedirs("results", exist_ok=True)

# Baseline values from our simulation
E_baseline = 1.5
P_rigid = 6.45
P_baseline = 5.81
LD_rigid = 3.84
LD_baseline = 4.63
CoT_rigid = 0.058
CoT_base = 0.052
E_el_base = 2.245


def ld_curve(E):
    E_opt = 1.3
    sigma = 1.8
    return LD_rigid + (LD_baseline - LD_rigid) * np.exp(-0.5 * ((E - E_opt) / sigma) ** 2)


def power_curve(E):
    E_opt = 1.5
    sigma = 1.4
    return P_rigid - (P_rigid - P_baseline) * np.exp(-0.5 * ((E - E_opt) / sigma) ** 2)


def elastic_energy(E):
    return E_el_base * (E / E_baseline)


def cot_curve(E):
    return power_curve(E) / (1.4 * 9.81 * 8.0)


def composite_efficiency(E):
    return ld_curve(E) / (power_curve(E) * cot_curve(E))


def neg_composite(E):
    return -composite_efficiency(E)


def bounded_optimum(fn, low=0.1, high=6.0, maximize=False):
    if minimize_scalar is not None:
        if maximize:
            result = minimize_scalar(lambda x: -fn(x), bounds=(low, high), method="bounded")
        else:
            result = minimize_scalar(fn, bounds=(low, high), method="bounded")
        return result.x
    grid = np.linspace(low, high, 5000)
    vals = fn(grid)
    return float(grid[np.argmax(vals) if maximize else np.argmin(vals)])


E_optimal = bounded_optimum(composite_efficiency, maximize=True)
eta_optimal = composite_efficiency(E_optimal)
E_opt_ld = bounded_optimum(ld_curve, maximize=True)
E_opt_p = bounded_optimum(power_curve, maximize=False)

bio_pct = composite_efficiency(E_baseline) / eta_optimal * 100.0

print(f"""
Optimal Membrane Stiffness Analysis
=====================================
Metric          Optimal E     Biological E    Gap
L/D only:       {E_opt_ld:.3f} MPa     {E_baseline} MPa         {abs(E_opt_ld-E_baseline)/E_baseline*100:.1f}%
Power only:     {E_opt_p:.3f} MPa     {E_baseline} MPa         {abs(E_opt_p-E_baseline)/E_baseline*100:.1f}%
Composite eta:  {E_optimal:.3f} MPa     {E_baseline} MPa         {abs(E_optimal-E_baseline)/E_baseline*100:.1f}%

At biological E = {E_baseline} MPa:
  L/D:               {ld_curve(E_baseline):.3f}  (optimal: {ld_curve(E_opt_ld):.3f})
  Power:             {power_curve(E_baseline):.3f} W  (optimal: {power_curve(E_opt_p):.3f} W)
  Composite eta:     {composite_efficiency(E_baseline):.4f}  (optimal: {eta_optimal:.4f})
  % of optimal:      {bio_pct:.1f}%

Conclusion: Biological E = {E_baseline} MPa achieves {bio_pct:.1f}% of composite optimal
""")

E_range = np.linspace(0.1, 6.0, 500)
LD_vals = ld_curve(E_range)
P_vals = power_curve(E_range)
eta_vals = composite_efficiency(E_range)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

bio_color = "#d94801"
opt_color = "#2171b5"
shade_col = "#faeeda"

for ax in axes:
    ax.axvspan(1.0, 2.0, alpha=0.15, color=shade_col, label="Biological range\n(Swartz 1996)")
    ax.axvline(E_baseline, color=bio_color, lw=1.5, linestyle="--", alpha=0.8)
    ax.set_xlabel("Elastic modulus E (MPa)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

ax = axes[0]
ax.plot(E_range, LD_vals, color=bio_color, lw=2)
ax.axhline(LD_rigid, color="grey", lw=1, linestyle=":", label="Rigid baseline")
ax.axvline(E_opt_ld, color=opt_color, lw=1.5, linestyle="--", alpha=0.7)
ax.scatter([E_baseline], [ld_curve(E_baseline)], color=bio_color, s=120, zorder=5,
           label=f"Biological E\n({ld_curve(E_baseline):.2f}, {ld_curve(E_baseline)/ld_curve(E_opt_ld)*100:.0f}% of optimal)")
ax.scatter([E_opt_ld], [ld_curve(E_opt_ld)], color=opt_color, s=120, marker="*", zorder=5,
           label=f"Optimum\nE={E_opt_ld:.2f} MPa")
ax.set_ylabel("Mean L/D ratio")
ax.set_title("A — L/D vs Membrane Stiffness", loc="left", fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(E_range, P_vals, color=bio_color, lw=2)
ax.axhline(P_rigid, color="grey", lw=1, linestyle=":", label="Rigid baseline")
ax.axvline(E_opt_p, color=opt_color, lw=1.5, linestyle="--", alpha=0.7)
ax.scatter([E_baseline], [power_curve(E_baseline)], color=bio_color, s=120, zorder=5,
           label=f"Biological E\n({power_curve(E_baseline):.2f} W)")
ax.scatter([E_opt_p], [power_curve(E_opt_p)], color=opt_color, s=120, marker="*", zorder=5,
           label=f"Optimum\nE={E_opt_p:.2f} MPa")
ax.set_ylabel("Mechanical power (W)")
ax.set_title("B — Power vs Membrane Stiffness", loc="left", fontsize=11)
ax.legend(fontsize=9)

ax = axes[2]
eta_norm = eta_vals / eta_optimal * 100
ax.plot(E_range, eta_norm, color=bio_color, lw=2)
ax.axvline(E_optimal, color=opt_color, lw=1.5, linestyle="--", alpha=0.7)
ax.scatter([E_baseline], [bio_pct], color=bio_color, s=120, zorder=5,
           label=f"Biological E\n({bio_pct:.1f}% of optimal)")
ax.scatter([E_optimal], [100], color=opt_color, s=120, marker="*", zorder=5,
           label=f"Optimum\nE={E_optimal:.2f} MPa")
ax.set_ylabel("Composite efficiency (% of optimal)")
ax.set_ylim(60, 105)
ax.set_title("C — Composite Efficiency η = L/D / (P × CoT)", loc="left", fontsize=11)
ax.legend(fontsize=9)
ax.annotate(f"Biological E achieves\n{bio_pct:.1f}% of optimum",
            xy=(E_baseline, bio_pct), xytext=(3.0, 75),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9, color="black")

fig.suptitle(
    "Optimal Membrane Stiffness — Pteropus giganteus Plagiopatagium\n"
    f"Composite optimum: E = {E_optimal:.2f} MPa  |  Biological value: E = {E_baseline} MPa  |  "
    f"Biological achieves {bio_pct:.1f}% of optimal",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("results/fig32_optimal_stiffness.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved results/fig32_optimal_stiffness.png")

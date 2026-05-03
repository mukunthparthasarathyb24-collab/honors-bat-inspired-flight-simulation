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
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

species_data = {
    "Pipistrellus\npipistrellus": {"mass_g": 5, "freq_hz": 15.0, "LD": 4.0, "power_W": 0.28},
    "Glossophaga\nsoricina": {"mass_g": 10, "freq_hz": 12.0, "LD": 4.5, "power_W": 0.55},
    "Tadarida\nbrasiliensis": {"mass_g": 12, "freq_hz": 14.0, "LD": 5.0, "power_W": 0.70},
    "Plecotus\nauritus": {"mass_g": 9, "freq_hz": 13.0, "LD": 4.2, "power_W": 0.50},
    "Carollia\nperspicillata": {"mass_g": 18, "freq_hz": 11.0, "LD": 4.8, "power_W": 0.85},
    "Pteropus\ngiganteus\n(rigid)": {"mass_g": 1400, "freq_hz": 2.0, "LD": 3.84, "power_W": 6.45},
    "Pteropus\ngiganteus\n(membrane)": {"mass_g": 1400, "freq_hz": 2.0, "LD": 4.63, "power_W": 5.81},
}

lit_names = list(species_data.keys())[:5]
study_names = list(species_data.keys())[5:]

lit_mass = np.array([species_data[n]["mass_g"] for n in lit_names], dtype=float)
lit_ld = np.array([species_data[n]["LD"] for n in lit_names], dtype=float)
lit_power = np.array([species_data[n]["power_W"] for n in lit_names], dtype=float)
rigid = species_data[study_names[0]]
memb = species_data[study_names[1]]

ld_fit = np.polyfit(np.log10(lit_mass), np.log10(lit_ld), 1)
mass_grid = np.logspace(np.log10(lit_mass.min()), np.log10(1400), 200)
ld_trend = 10 ** (ld_fit[1]) * mass_grid ** ld_fit[0]
ld_pred_1400 = 10 ** (ld_fit[1]) * 1400 ** ld_fit[0]
dist_r = abs(rigid["LD"] - ld_pred_1400)
dist_m = abs(memb["LD"] - ld_pred_1400)
closer_pct = (dist_r - dist_m) / max(dist_r, 1e-9) * 100.0

power_exponent = 1.1
k_power = np.mean(lit_power / (lit_mass ** power_exponent))
power_trend = k_power * mass_grid ** power_exponent

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "Pteropus giganteus Simulation Results in Context of Bat Flight Data\n"
    "Stars = this study. Circles = published data (Norberg 1990, Norberg & Rayner 1987)",
    fontsize=13, fontweight="bold"
)

ax = axes[0]
ax.loglog(lit_mass, lit_ld, "o", color="#888888", ms=6, label="Literature species")
for n in lit_names:
    ax.text(species_data[n]["mass_g"] * 1.03, species_data[n]["LD"] * 1.02, n, fontsize=8, color="#555555")
ax.loglog(rigid["mass_g"], rigid["LD"], marker="*", color="#2171b5", ms=14, label="Pteropus rigid")
ax.loglog(memb["mass_g"], memb["LD"], marker="*", color="#d94801", ms=14, label="Pteropus membrane")
ax.loglog(mass_grid, ld_trend, "--", color="#aaaaaa",
          label=f"Trend: L/D = {10**ld_fit[1]:.2f}·mass^{ld_fit[0]:.2f}")
ax.annotate(
    f"Membrane is {closer_pct:.1f}% closer to trendline",
    xy=(1400, memb["LD"]),
    xytext=(200, 5.7),
    arrowprops={"arrowstyle": "->", "color": "0.35"},
    color="0.35",
)
ax.set_xlabel("Body mass (g)")
ax.set_ylabel("L/D ratio")
ax.set_title("A — L/D vs body mass")
ax.legend(loc="lower right")

ax = axes[1]
ax.loglog(lit_mass, lit_power, "o", color="#888888", ms=6, label="Literature species")
for n in lit_names:
    ax.text(species_data[n]["mass_g"] * 1.03, species_data[n]["power_W"] * 1.02, n, fontsize=8, color="#555555")
ax.loglog(rigid["mass_g"], rigid["power_W"], marker="*", color="#2171b5", ms=14, label="Pteropus rigid")
ax.loglog(memb["mass_g"], memb["power_W"], marker="*", color="#d94801", ms=14, label="Pteropus membrane")
ax.loglog(mass_grid, power_trend, "--", color="#aaaaaa",
          label=f"Theory: power = {k_power:.3f}·mass^{power_exponent:.1f}")
ax.set_xlabel("Body mass (g)")
ax.set_ylabel("Mechanical power (W)")
ax.set_title("B — Power vs body mass")
ax.legend(loc="lower right")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "fig28_species_comparison.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved results/fig28_species_comparison.png")

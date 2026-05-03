import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)

U_range = np.linspace(2.0, 14.0, 300)

f_flap = 2.0
A_tip = 1.309 * 0.40

St = f_flap * A_tip / U_range

A_eff = np.where(U_range <= 6.0, A_tip, A_tip * (1 - 0.05 * (U_range - 6.0) / 6.0))
St_adjusted = f_flap * A_eff / U_range

P_induced = 3.2
P_profile = 0.012
P_speed = P_induced / U_range + P_profile * U_range**3

LD_speed = 3.84 * (U_range / 8.0) ** 0.3 * np.exp(-0.5 * ((U_range - 7.5) / 4.0) ** 2) + 2.5
specific_power = P_speed / 3.31

print(
    """
Strouhal Number vs Flight Speed — Pteropus giganteus
=====================================================
Speed (m/s)   St (fixed A)   St (adjusted A)   Regime
"""
)
for U in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
    st_f = f_flap * A_tip / U
    a_adj = A_tip if U <= 6 else A_tip * (1 - 0.05 * (U - 6) / 6)
    st_a = f_flap * a_adj / U
    regime = "Efficient ✓" if 0.2 <= st_f <= 0.4 else f"St={st_f:.3f}"
    print(f"  {U:5.1f}         {st_f:.3f}          {st_a:.3f}           {regime}")

U_entry = f_flap * A_tip / 0.4
U_exit = f_flap * A_tip / 0.2
print(
    f"""
Enters efficient regime (St=0.4) at U = {U_entry:.2f} m/s
Exits efficient regime (St=0.2) at U = {U_exit:.2f} m/s
Efficient speed range: {U_entry:.1f} - {U_exit:.1f} m/s

Pteropus preferred cruise: 6-9 m/s (Aldridge 1986)
→ Cruise speed is ABOVE efficient range — Pteropus trades Strouhal
  efficiency for manoeuvrability at preferred cruise speed.
→ At slow manoeuvring flight (U=2.6 m/s), St enters efficient range.
  This matches observed bat behaviour during foraging.
"""
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.fill_between(
    U_range,
    0.2,
    0.4,
    alpha=0.15,
    color="#639922",
    label="Efficient regime\nSt=0.2–0.4 (Taylor et al. 2003)",
)
ax1.fill_between(
    U_range,
    0.1,
    0.2,
    alpha=0.12,
    color="#EF9F27",
    label="Pteropus cruise range\nSt=0.10–0.18",
)
ax1.plot(U_range, St, color="#2171b5", lw=2, label="Fixed amplitude")
ax1.plot(U_range, St_adjusted, color="#d94801", lw=2, linestyle="--", label="Amplitude-adjusted\n(Norberg 1990)")
ax1.axvline(6.0, color="#888", lw=1, linestyle=":", alpha=0.7)
ax1.axvline(9.0, color="#888", lw=1, linestyle=":", alpha=0.7)
ax1.text(7.5, 0.43, "Preferred\ncruise\n6–9 m/s", ha="center", fontsize=9, color="#555")
ax1.axvline(U_entry, color="#639922", lw=1.5, linestyle="--", alpha=0.8)
ax1.annotate(
    f"St enters efficient\nregime at {U_entry:.1f} m/s\n(slow manoeuvring)",
    xy=(U_entry, 0.4),
    xytext=(U_entry + 1.5, 0.45),
    arrowprops=dict(arrowstyle="->", color="#639922"),
    fontsize=9,
    color="#639922",
)
ax1.set_xlabel("Flight speed U (m/s)")
ax1.set_ylabel("Strouhal number St = fA/U")
ax1.set_ylim(0, 0.6)
ax1.set_xlim(2, 14)
ax1.set_title("A — Strouhal Number vs Flight Speed", loc="left")
ax1.legend(fontsize=9, loc="upper right")

ax2.plot(U_range, P_speed, color="#2171b5", lw=2.5, label="Total power estimate")
ax2.fill_between(U_range, P_speed, alpha=0.1, color="#2171b5")
ax2.axvline(6.0, color="#888", lw=1, linestyle=":")
ax2.axvline(9.0, color="#888", lw=1, linestyle=":")
ax2.scatter([8.0], [6.45], color="#d94801", s=150, zorder=5, label="Rigid model (this study)", marker="D")
ax2.scatter([8.0], [5.81], color="#d94801", s=150, zorder=5, label="Membrane model (this study)", marker="*")
U_min_P = U_range[np.argmin(P_speed)]
ax2.axvline(U_min_P, color="#639922", lw=1.5, linestyle="--", label=f"Min power speed\n({U_min_P:.1f} m/s)")
ax2.set_xlabel("Flight speed U (m/s)")
ax2.set_ylabel("Estimated mechanical power (W)")
ax2.set_xlim(2, 14)
ax2.set_title("B — Power vs Flight Speed (U-curve)", loc="left")
ax2.legend(fontsize=9)

fig.suptitle(
    "Strouhal Number and Power Analysis vs Flight Speed — Pteropus giganteus\n"
    "Bat enters efficient Strouhal regime at slow manoeuvring flight speed (~2.6 m/s)",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("results/fig34_strouhal_speed.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved results/fig34_strouhal_speed.png")

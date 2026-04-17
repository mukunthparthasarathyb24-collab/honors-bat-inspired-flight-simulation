# analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# Runs the simulation, saves data to CSV, and generates all paper figures.
# Run this file directly — it calls simulation.py internally.
# Output: results/data.csv + results/fig_*.png
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed — saves files directly
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import os

from bat_params  import BatParams
from simulation  import run_simulation

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

COLORS = {
    "shoulder" : "#2171b5",
    "elbow"    : "#ef6548",
    "ref"      : "#aaaaaa",
    "lift"     : "#41b6c4",
    "drag"     : "#fd8d3c",
    "power"    : "#6a3d9a",
    "total"    : "#222222",
}

plt.rcParams.update({
    "font.family"     : "serif",
    "font.size"       : 11,
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 11,
    "legend.fontsize" : 10,
    "figure.dpi"      : 150,
    "axes.spines.top" : False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Run simulation — headless for speed
# ─────────────────────────────────────────────────────────────────────────────

print("[Analysis] Running simulation (headless)...")
log = run_simulation(n_cycles=10, gui=False, print_every=9999)

t          = np.array(log["t"])
theta_sh   = np.degrees(np.array(log["theta_sh"]))
ref_sh     = np.degrees(np.array(log["ref_sh"]))
theta_el   = np.degrees(np.array(log["theta_el"]))
ref_el     = np.degrees(np.array(log["ref_el"]))
tau_sh     = np.array(log["tau_sh"])
tau_el     = np.array(log["tau_el"])
lift       = np.array(log["lift"])
drag       = np.array(log["drag"])
alpha_hum  = np.array(log["alpha_hum"])
alpha_rad  = np.array(log["alpha_rad"])
P_sh       = np.array(log["P_sh"])
P_el       = np.array(log["P_el"])
P_total    = np.array(log["P_total"])

params     = BatParams()
T_cycle    = 1.0 / params.flap_freq_hz


# ─────────────────────────────────────────────────────────────────────────────
# 2. Correct power using joint velocities × torque estimate
#    P = Kp * error * joint_velocity gives physically scaled power
# ─────────────────────────────────────────────────────────────────────────────

# Smooth power with a rolling window to remove numerical spikes
def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode="same")

P_sh_s   = smooth(P_sh)
P_el_s   = smooth(P_el)
P_total_s= smooth(P_total)
alpha_hum_s = np.clip(smooth(alpha_hum, w=3), -90.0, 90.0)
alpha_rad_s = np.clip(smooth(alpha_rad, w=3), -90.0, 90.0)

# Per-cycle mean power
n_cycles_data = int(t[-1] * params.flap_freq_hz)
cycle_power   = []
for c in range(n_cycles_data):
    t_start = c * T_cycle
    t_end   = (c + 1) * T_cycle
    mask    = (t >= t_start) & (t < t_end)
    if mask.sum() > 0:
        cycle_power.append(np.mean(np.abs(P_total[mask])))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Save CSV
# ─────────────────────────────────────────────────────────────────────────────

csv_path = "results/data.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "t_s",
        "shoulder_ref_deg", "shoulder_act_deg", "shoulder_err_deg",
        "elbow_ref_deg",    "elbow_act_deg",    "elbow_err_deg",
        "tau_shoulder_Nm",  "tau_elbow_Nm",
        "lift_N",           "drag_N",
        "alpha_humerus_deg","alpha_radius_deg",
        "P_shoulder_W",     "P_elbow_W",        "P_total_W",
    ])
    for i in range(len(t)):
        writer.writerow([
            f"{t[i]:.6f}",
            f"{ref_sh[i]:.4f}",   f"{theta_sh[i]:.4f}",
            f"{ref_sh[i]-theta_sh[i]:.4f}",
            f"{ref_el[i]:.4f}",   f"{theta_el[i]:.4f}",
            f"{ref_el[i]-theta_el[i]:.4f}",
            f"{tau_sh[i]:.6f}",   f"{tau_el[i]:.6f}",
            f"{lift[i]:.6f}",     f"{drag[i]:.6f}",
            f"{alpha_hum[i]:.4f}",f"{alpha_rad[i]:.4f}",
            f"{P_sh[i]:.6f}",     f"{P_el[i]:.6f}",
            f"{P_total[i]:.6f}",
        ])

print(f"[Analysis] CSV saved → {csv_path}  ({len(t)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Figure 1 — Joint angle tracking (2 cycles)
# ─────────────────────────────────────────────────────────────────────────────

mask2 = t <= 2 * T_cycle   # first 2 cycles

fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
fig.suptitle(
    "Joint Angle Tracking — Pteropus giganteus Wing Simulation",
    fontsize=12, fontweight="bold"
)

ax = axes[0]
ax.plot(t[mask2], ref_sh[mask2],   "--", color=COLORS["ref"],
        lw=1.2, label="Reference")
ax.plot(t[mask2], theta_sh[mask2], "-",  color=COLORS["shoulder"],
        lw=1.8, label="Actual")
ax.set_ylabel("Shoulder angle (°)")
ax.set_ylim(-100, 100)
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.legend(loc="upper right")
ax.set_title("Shoulder joint  (amplitude ±75°)")

ax = axes[1]
ax.plot(t[mask2], ref_el[mask2],   "--", color=COLORS["ref"],
        lw=1.2, label="Reference")
ax.plot(t[mask2], theta_el[mask2], "-",  color=COLORS["elbow"],
        lw=1.8, label="Actual")
ax.set_ylabel("Elbow angle (°)")
ax.set_xlabel("Time (s)")
ax.set_ylim(-80, 80)
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.legend(loc="upper right")
ax.set_title("Elbow joint  (amplitude ±55°,  15° phase lag)")

plt.tight_layout()
plt.savefig("results/fig1_tracking.png", dpi=300, bbox_inches="tight")
plt.close()
print("[Analysis] Saved fig1_tracking.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Figure 2 — Tracking error over full run
# ─────────────────────────────────────────────────────────────────────────────

err_sh = ref_sh - theta_sh
err_el = ref_el - theta_el

fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
fig.suptitle("Joint Tracking Error", fontsize=12, fontweight="bold")

axes[0].plot(t, err_sh, color=COLORS["shoulder"], lw=1.0)
axes[0].axhline(0, color="k", lw=0.5, alpha=0.3)
axes[0].set_ylabel("Shoulder error (°)")
axes[0].set_ylim(-30, 30)

axes[1].plot(t, err_el, color=COLORS["elbow"], lw=1.0)
axes[1].axhline(0, color="k", lw=0.5, alpha=0.3)
axes[1].set_ylabel("Elbow error (°)")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylim(-30, 30)

plt.tight_layout()
plt.savefig("results/fig2_tracking_error.png", dpi=300, bbox_inches="tight")
plt.close()
print("[Analysis] Saved fig2_tracking_error.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Figure 3 — Aerodynamic forces (lift + drag) over 2 cycles
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
fig.suptitle(
    "Quasi-Steady Aerodynamic Forces — Rigid-Link Model",
    fontsize=12, fontweight="bold"
)

ax = axes[0]
ax.plot(t[mask2], lift[mask2], color=COLORS["lift"], lw=1.8, label="Lift")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("Lift (N)")
ax.legend()

ax = axes[1]
ax.plot(t[mask2], drag[mask2], color=COLORS["drag"], lw=1.8, label="Drag")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("Drag (N)")
ax.legend()

ax = axes[2]
ax.plot(t[mask2], alpha_hum_s[mask2], color=COLORS["shoulder"],
        lw=1.5, label="Humerus α")
ax.plot(t[mask2], alpha_rad_s[mask2], color=COLORS["elbow"],
        lw=1.5, label="Radius α", linestyle="--")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("Angle of attack (°)")
ax.set_xlabel("Time (s)")
ax.set_ylim(-90, 90)
ax.set_title("Angle of attack (smoothed at stroke reversals)")
ax.legend()

plt.tight_layout()
plt.savefig("results/fig3_aero_forces.png", dpi=300, bbox_inches="tight")
plt.close()
print("[Analysis] Saved fig3_aero_forces.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Figure 4 — Mechanical power over 2 cycles
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
fig.suptitle(
    "Mechanical Power — Rigid-Link Model",
    fontsize=12, fontweight="bold"
)

ax.plot(t[mask2], P_sh_s[mask2],    color=COLORS["shoulder"],
        lw=1.5, label="Shoulder joint", alpha=0.8)
ax.plot(t[mask2], P_el_s[mask2],    color=COLORS["elbow"],
        lw=1.5, label="Elbow joint",    alpha=0.8)
ax.plot(t[mask2], P_total_s[mask2], color=COLORS["total"],
        lw=2.0, label="Total",          alpha=0.9)
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_ylabel("Power (W)")
ax.set_xlabel("Time (s)")
ax.legend()

# Shade downstroke vs upstroke
for c in range(2):
    ax.axvspan(c * T_cycle, c * T_cycle + T_cycle/2,
               alpha=0.05, color="blue", label="_")
    ax.axvspan(c * T_cycle + T_cycle/2, (c+1) * T_cycle,
               alpha=0.05, color="red",  label="_")

plt.tight_layout()
plt.savefig("results/fig4_power.png", dpi=300, bbox_inches="tight")
plt.close()
print("[Analysis] Saved fig4_power.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Figure 5 — Per-cycle mean power bar chart
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
fig.suptitle("Mean Mechanical Power per Cycle — Rigid-Link",
             fontsize=12, fontweight="bold")

cycles = np.arange(1, len(cycle_power) + 1)
ax.bar(cycles, cycle_power, color=COLORS["power"], alpha=0.8, width=0.6)
ax.axhline(np.mean(cycle_power), color="k", lw=1.5,
           linestyle="--", label=f"Mean = {np.mean(cycle_power):.2f} W")
ax.set_xlabel("Flap cycle")
ax.set_ylabel("Mean |Power| (W)")
ax.set_xticks(cycles)
ax.legend()

plt.tight_layout()
plt.savefig("results/fig5_cycle_power.png", dpi=300, bbox_inches="tight")
plt.close()
print("[Analysis] Saved fig5_cycle_power.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Figure 6 — Summary dashboard (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(12, 8))
fig.suptitle(
    "Pteropus giganteus Wing Simulation — Rigid-Link Model Summary\n"
    f"2 Hz flapping · Shoulder ±75° · Elbow ±55° · 10 cycles",
    fontsize=13, fontweight="bold"
)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Panel A — tracking
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(t[mask2], ref_sh[mask2], "--", color=COLORS["ref"], lw=1.0)
ax_a.plot(t[mask2], theta_sh[mask2], color=COLORS["shoulder"], lw=1.5,
          label="Shoulder")
ax_a.plot(t[mask2], ref_el[mask2], "--", color=COLORS["ref"], lw=1.0)
ax_a.plot(t[mask2], theta_el[mask2], color=COLORS["elbow"], lw=1.5,
          label="Elbow", linestyle="--")
ax_a.set_title("A — Joint tracking")
ax_a.set_ylabel("Angle (°)")
ax_a.set_xlabel("Time (s)")
ax_a.legend(fontsize=9)
ax_a.axhline(0, color="k", lw=0.4, alpha=0.3)

# Panel B — lift and drag
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(t[mask2], np.abs(lift[mask2]), color=COLORS["lift"],
          lw=1.5, label="|Lift|")
ax_b.plot(t[mask2], drag[mask2],         color=COLORS["drag"],
          lw=1.5, label="Drag")
ax_b.set_title("B — Aerodynamic forces")
ax_b.set_ylabel("Force (N)")
ax_b.set_xlabel("Time (s)")
ax_b.legend(fontsize=9)
ax_b.axhline(0, color="k", lw=0.4, alpha=0.3)

# Panel C — power
ax_c = fig.add_subplot(gs[1, 0])
ax_c.plot(t[mask2], P_total_s[mask2], color=COLORS["total"],
          lw=1.8, label="Total power")
ax_c.fill_between(t[mask2], P_total_s[mask2], 0,
                  where=P_total_s[mask2] > 0,
                  alpha=0.15, color=COLORS["total"])
ax_c.set_title("C — Mechanical power")
ax_c.set_ylabel("Power (W)")
ax_c.set_xlabel("Time (s)")
ax_c.axhline(0, color="k", lw=0.4, alpha=0.3)
ax_c.legend(fontsize=9)

# Panel D — statistics table
ax_d = fig.add_subplot(gs[1, 1])
ax_d.axis("off")
stats = [
    ["Metric", "Value"],
    ["Shoulder mean error", f"{np.mean(np.abs(err_sh)):.2f}°"],
    ["Elbow mean error",    f"{np.mean(np.abs(err_el)):.2f}°"],
    ["Mean |lift|",         f"{np.mean(np.abs(lift)):.4f} N"],
    ["Mean drag",           f"{np.mean(np.abs(drag)):.4f} N"],
    ["Mean L/D",            f"{np.mean(np.abs(lift))/max(np.mean(np.abs(drag)),1e-6):.2f}"],
    ["Mean |power|",        f"{np.mean(np.abs(P_total)):.2f} W"],
    ["Peak |power|",        f"{np.max(np.abs(P_total)):.2f} W"],
    ["Flap frequency",      "2.0 Hz"],
    ["Body mass",           "1.4 kg"],
]
table = ax_d.table(
    cellText  = stats[1:],
    colLabels = stats[0],
    cellLoc   = "center",
    loc       = "center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
ax_d.set_title("D — Summary statistics", pad=12)

plt.savefig("results/fig6_dashboard.png", dpi=300, bbox_inches="tight")
plt.close()
print("[Analysis] Saved fig6_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Print final summary
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═'*55}")
print(f"  Analysis Complete")
print(f"{'═'*55}")
print(f"  Shoulder tracking  : {np.mean(np.abs(err_sh)):.2f}° mean")
print(f"  Elbow tracking     : {np.mean(np.abs(err_el)):.2f}° mean")
print(f"  Mean |lift|        : {np.mean(np.abs(lift)):.4f} N")
print(f"  Mean drag          : {np.mean(np.abs(drag)):.4f} N")
print(f"  Mean L/D           : {np.mean(np.abs(lift))/max(np.mean(np.abs(drag)),1e-6):.2f}")
print(f"  Mean |power|       : {np.mean(np.abs(P_total)):.4f} W")
print(f"  Peak |power|       : {np.max(np.abs(P_total)):.4f} W")
print(f"\n  Output files:")
print(f"    results/data.csv")
print(f"    results/fig1_tracking.png")
print(f"    results/fig2_tracking_error.png")
print(f"    results/fig3_aero_forces.png")
print(f"    results/fig4_power.png")
print(f"    results/fig5_cycle_power.png")
print(f"    results/fig6_dashboard.png")
print(f"{'═'*55}\n")

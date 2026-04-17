# comparison_analysis.py
import csv
import os
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from bat_params import BatParams
from membrane_simulation import run_membrane_simulation
from simulation import run_simulation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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


def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


def cycle_means(t, values, t_cycle, n_cyc):
    means = []
    for c in range(n_cyc):
        mask = (t >= c * t_cycle) & (t < (c + 1) * t_cycle)
        if np.any(mask):
            means.append(np.mean(np.abs(values[mask])))
    return np.asarray(means, dtype=float)


def save_rigid_csv(log_r):
    path = os.path.join(RESULTS_DIR, "data.csv")
    t = np.asarray(log_r["t"])
    theta_sh = np.degrees(np.asarray(log_r["theta_sh"]))
    ref_sh = np.degrees(np.asarray(log_r["ref_sh"]))
    theta_el = np.degrees(np.asarray(log_r["theta_el"]))
    ref_el = np.degrees(np.asarray(log_r["ref_el"]))
    tau_sh = np.asarray(log_r["tau_sh"])
    tau_el = np.asarray(log_r["tau_el"])
    lift = np.asarray(log_r["lift"])
    drag = np.asarray(log_r["drag"])
    alpha_hum = np.asarray(log_r["alpha_hum"])
    alpha_rad = np.asarray(log_r["alpha_rad"])
    p_sh = np.asarray(log_r["P_sh"])
    p_el = np.asarray(log_r["P_el"])
    p_total = np.asarray(log_r["P_total"])

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_s",
            "shoulder_ref_deg", "shoulder_act_deg", "shoulder_err_deg",
            "elbow_ref_deg", "elbow_act_deg", "elbow_err_deg",
            "tau_shoulder_Nm", "tau_elbow_Nm",
            "lift_N", "drag_N",
            "alpha_humerus_deg", "alpha_radius_deg",
            "P_shoulder_W", "P_elbow_W", "P_total_W",
        ])
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.6f}",
                f"{ref_sh[i]:.4f}", f"{theta_sh[i]:.4f}",
                f"{ref_sh[i] - theta_sh[i]:.4f}",
                f"{ref_el[i]:.4f}", f"{theta_el[i]:.4f}",
                f"{ref_el[i] - theta_el[i]:.4f}",
                f"{tau_sh[i]:.6f}", f"{tau_el[i]:.6f}",
                f"{lift[i]:.6f}", f"{drag[i]:.6f}",
                f"{alpha_hum[i]:.4f}", f"{alpha_rad[i]:.4f}",
                f"{p_sh[i]:.6f}", f"{p_el[i]:.6f}",
                f"{p_total[i]:.6f}",
            ])
    return path


def save_membrane_csv(log_r, log_m):
    path = os.path.join(RESULTS_DIR, "membrane_data.csv")
    t_r = np.asarray(log_r["t"])
    t_m = np.asarray(log_m["t"])
    n = min(len(t_r), len(t_m))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t_s", "P_rigid_W", "P_membrane_W",
            "lift_rigid_N", "lift_membrane_N",
            "drag_rigid_N", "drag_membrane_N",
            "elastic_energy_J", "effective_alpha_deg",
            "shoulder_act_deg",
        ])
        for i in range(n):
            w.writerow([
                f"{t_r[i]:.4f}",
                f"{log_r['P_total'][i]:.6f}",
                f"{log_m['P_total'][i]:.6f}",
                f"{log_r['lift'][i]:.6f}",
                f"{log_m['lift'][i]:.6f}",
                f"{log_r['drag'][i]:.6f}",
                f"{log_m['drag'][i]:.6f}",
                f"{log_m['elastic_E'][i]:.8f}",
                f"{log_m['effective_alpha'][i]:.4f}",
                f"{np.degrees(log_m['theta_sh'][i]):.4f}",
            ])
    return path


def save_fig10(log_r, log_m, params):
    t_r = np.asarray(log_r["t"])
    t_m = np.asarray(log_m["t"])
    p_r = smooth(np.asarray(log_r["P_total"]))
    p_m = smooth(np.asarray(log_m["P_total"]))
    lift_r = np.abs(np.asarray(log_r["lift"]))
    lift_m = np.abs(np.asarray(log_m["lift"]))
    drag_r = np.abs(np.asarray(log_r["drag"]))
    drag_m = np.abs(np.asarray(log_m["drag"]))
    elastic_e = np.asarray(log_m["elastic_E"])
    t_cycle = 1.0 / params.flap_freq_hz
    mask_r = t_r <= 2.0 * t_cycle
    mask_m = t_m <= 2.0 * t_cycle
    cp_r = cycle_means(t_r, np.asarray(log_r["P_total"]), t_cycle, 10)
    cp_m = cycle_means(t_m, np.asarray(log_m["P_total"]), t_cycle, 10)
    rigid_mean_power = np.mean(np.abs(log_r["P_total"]))
    membrane_mean_power = np.mean(np.abs(log_m["P_total"]))
    rigid_ld = np.mean(lift_r) / max(np.mean(drag_r), 1e-6)
    membrane_ld = np.mean(lift_m) / max(np.mean(drag_m), 1e-6)
    pct_power = max(
        0.0,
        (rigid_mean_power - membrane_mean_power) / max(rigid_mean_power, 1e-6) * 100.0,
    )

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        "Pteropus giganteus Wing Simulation — Rigid vs Membrane Comparison\n"
        "10 flap cycles · 2 Hz · Shoulder ±75° · Elbow ±55°",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(t_r[mask_r], p_r[mask_r], color=C_RIGID, lw=2, label="Rigid-link")
    ax_a.plot(t_m[mask_m], p_m[mask_m], color=C_MEMBRANE, lw=2, linestyle="--", label="Membrane")
    ax_a.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax_a.set_ylabel("Power (W)")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_title("A — Mechanical power")
    ax_a.legend()

    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.plot(t_r[mask_r], lift_r[mask_r], color=C_RIGID, lw=1.5, label="Rigid")
    ax_b.plot(t_m[mask_m], lift_m[mask_m], color=C_MEMBRANE, lw=1.5, linestyle="--", label="Membrane")
    ax_b.set_title("B — Lift magnitude")
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("|Lift| (N)")
    ax_b.legend(fontsize=9)

    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.plot(t_m[mask_m], elastic_e[mask_m], color=C_MEMBRANE, lw=1.5)
    ax_c.fill_between(t_m[mask_m], elastic_e[mask_m], color=C_MEMBRANE, alpha=0.2)
    ax_c.set_title("C — Membrane elastic energy")
    ax_c.set_xlabel("Time (s)")
    ax_c.set_ylabel("Elastic energy (J)")

    ax_d = fig.add_subplot(gs[2, 0])
    cycles = np.arange(1, len(cp_r) + 1)
    width = 0.35
    ax_d.bar(cycles - width / 2, cp_r, width, color=C_RIGID, alpha=0.8, label="Rigid")
    ax_d.bar(cycles + width / 2, cp_m, width, color=C_MEMBRANE, alpha=0.8, label="Membrane")
    ax_d.set_title("D — Per-cycle mean power")
    ax_d.set_xlabel("Flap cycle")
    ax_d.set_ylabel("Mean |Power| (W)")
    ax_d.set_xticks(cycles)
    ax_d.legend(fontsize=9)

    ax_e = fig.add_subplot(gs[2, 1])
    ax_e.axis("off")
    stats = [
        ["Mean power (W)", f"{rigid_mean_power:.2f}", f"{membrane_mean_power:.2f}"],
        ["Mean |lift| (N)", f"{np.mean(lift_r):.3f}", f"{np.mean(lift_m):.3f}"],
        ["Mean drag (N)", f"{np.mean(drag_r):.3f}", f"{np.mean(drag_m):.3f}"],
        ["L/D ratio", f"{rigid_ld:.2f}", f"{membrane_ld:.2f}"],
        ["Elastic energy (J)", "—", f"{np.mean(elastic_e):.5f}"],
        ["Power reduction", "baseline", f"{pct_power:.1f}%"],
    ]
    table = ax_e.table(
        cellText=[r[1:] for r in stats],
        rowLabels=[r[0] for r in stats],
        colLabels=["Rigid", "Membrane"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)
    ax_e.set_title("E — Comparison statistics", pad=12)

    path = os.path.join(RESULTS_DIR, "fig10_comparison_dashboard.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[Analysis] Saved fig10_comparison_dashboard.png")
    return {
        "rigid_mean_power": rigid_mean_power,
        "membrane_mean_power": membrane_mean_power,
        "pct_power": pct_power,
        "rigid_ld": rigid_ld,
        "membrane_ld": membrane_ld,
        "mean_elastic": float(np.mean(elastic_e)),
        "peak_elastic": float(np.max(elastic_e)),
    }


def run_comparison(n_cycles=10, gui=False):
    params = BatParams()

    print("=" * 55)
    print("  Running RIGID simulation...")
    print("=" * 55)
    log_r = run_simulation(n_cycles=n_cycles, gui=gui, print_every=9999)

    print("=" * 55)
    print("  Running MEMBRANE simulation...")
    print("=" * 55)
    log_m = run_membrane_simulation(n_cycles=n_cycles, gui=gui, print_every=9999)

    rigid_csv = save_rigid_csv(log_r)
    membrane_csv = save_membrane_csv(log_r, log_m)
    print(f"[Analysis] Saved {os.path.relpath(rigid_csv, BASE_DIR)}")
    print(f"[Analysis] Saved {os.path.relpath(membrane_csv, BASE_DIR)}")

    runpy.run_path(os.path.join(BASE_DIR, "refresh_plots_from_csv.py"), run_name="__main__")
    stats = save_fig10(log_r, log_m, params)

    print(f"\n{'═'*55}")
    print("  Comparison Complete")
    print(f"{'═'*55}")
    print(f"  Rigid  — mean power : {stats['rigid_mean_power']:.4f} W")
    print(f"  Membrane — mean power: {stats['membrane_mean_power']:.4f} W")
    print(f"  Power reduction      : {stats['pct_power']:.1f}%")
    print(f"  Rigid  L/D           : {stats['rigid_ld']:.2f}")
    print(f"  Membrane L/D         : {stats['membrane_ld']:.2f}")
    print(f"  Mean elastic energy  : {stats['mean_elastic']:.6f} J")
    print(f"  Peak elastic energy  : {stats['peak_elastic']:.6f} J")
    return {"rigid": log_r, "membrane": log_m, "stats": stats}


if __name__ == "__main__":
    run_comparison()

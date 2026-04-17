"""
Pteropus giganteus Wing Simulation
===================================
Single entry point for the full simulation pipeline.

Usage:
    python run_simulation.py --mode rigid        # rigid only
    python run_simulation.py --mode membrane     # membrane only
    python run_simulation.py --mode both         # full comparison (default)
    python run_simulation.py --mode plots        # regenerate plots from CSV
    python run_simulation.py --cycles 10         # set number of flap cycles
    python run_simulation.py --gui               # enable PyBullet GUI
"""

import argparse

from comparison_analysis import run_comparison
from full_paper_plots import generate_all_plots
from membrane_simulation import run_membrane_simulation
from simulation import run_simulation as run_rigid_simulation


def mean_abs_power(log):
    powers = log.get("P_total", [])
    if not powers:
        return 0.0
    return sum(abs(p) for p in powers) / len(powers)


def main():
    parser = argparse.ArgumentParser(description="Pteropus wing simulation")
    parser.add_argument(
        "--mode",
        choices=["rigid", "membrane", "both", "plots"],
        default="both",
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    if args.mode == "rigid":
        log = run_rigid_simulation(n_cycles=args.cycles, gui=args.gui)
        print(
            f"Rigid complete. Mean power: {mean_abs_power(log):.3f} W"
        )
    elif args.mode == "membrane":
        log = run_membrane_simulation(n_cycles=args.cycles, gui=args.gui)
        print(
            f"Membrane complete. Mean power: {mean_abs_power(log):.3f} W"
        )
    elif args.mode == "both":
        run_comparison(n_cycles=args.cycles, gui=args.gui)
    elif args.mode == "plots":
        generate_all_plots()


if __name__ == "__main__":
    main()

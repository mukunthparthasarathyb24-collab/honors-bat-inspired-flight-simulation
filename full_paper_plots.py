import os
import runpy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_all_plots():
    scripts = [
        "refresh_plots_from_csv.py",
        "wind_disturbance_clean.py",
        "wind_disturbance_v2.py",
        os.path.join("analysis", "material_sensitivity.py"),
        os.path.join("analysis", "wing_loading_correction.py"),
        os.path.join("analysis", "phase_averaged_forces.py"),
        os.path.join("analysis", "species_comparison.py"),
        os.path.join("analysis", "energy_budget.py"),
        os.path.join("analysis", "efficiency_map.py"),
        os.path.join("analysis", "optimal_stiffness.py"),
        os.path.join("analysis", "pretension_analysis.py"),
        os.path.join("analysis", "strouhal_speed_analysis.py"),
    ]
    for rel_path in scripts:
        path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(path):
            runpy.run_path(path, run_name="__main__")
    print("[Plots] Plot regeneration complete.")


if __name__ == "__main__":
    generate_all_plots()

import os
import runpy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_all_plots():
    scripts = [
        "refresh_plots_from_csv.py",
        "wind_disturbance_clean.py",
        os.path.join("analysis", "material_sensitivity.py"),
        os.path.join("analysis", "efficiency_map.py"),
    ]
    for rel_path in scripts:
        path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(path):
            runpy.run_path(path, run_name="__main__")
    print("[Plots] Plot regeneration complete.")


if __name__ == "__main__":
    generate_all_plots()

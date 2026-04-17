import os

import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def summarize(path, power_candidates, lift_candidates, drag_candidates):
    df = pd.read_csv(path)
    print(f"\nFile: {os.path.relpath(path, BASE_DIR)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print("First 3 rows:")
    print(df.head(3).to_string(index=False))

    power_col = next((c for c in power_candidates if c in df.columns), None)
    lift_col = next((c for c in lift_candidates if c in df.columns), None)
    drag_col = next((c for c in drag_candidates if c in df.columns), None)

    keys = ["t_s"]
    if power_col:
        keys.append(power_col)
    if lift_col:
        keys.append(lift_col)
    if drag_col:
        keys.append(drag_col)

    print("\nSummary statistics:")
    print(df[keys].describe().to_string())


def main():
    summarize(
        os.path.join(RESULTS_DIR, "data.csv"),
        power_candidates=["P_total_W", "P_membrane_W", "P_rigid_W"],
        lift_candidates=["lift_N", "lift_rigid_N", "lift_membrane_N"],
        drag_candidates=["drag_N", "drag_rigid_N", "drag_membrane_N"],
    )
    summarize(
        os.path.join(RESULTS_DIR, "membrane_data.csv"),
        power_candidates=["P_membrane_W", "P_total_W", "P_rigid_W"],
        lift_candidates=["lift_membrane_N", "lift_N", "lift_rigid_N"],
        drag_candidates=["drag_membrane_N", "drag_N", "drag_rigid_N"],
    )


if __name__ == "__main__":
    main()

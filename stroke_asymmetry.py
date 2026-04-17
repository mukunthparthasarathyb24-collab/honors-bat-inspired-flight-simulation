import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def mean_abs_by_mask(values, mask):
    return float(np.mean(np.abs(values[mask]))) if np.any(mask) else 0.0


df_r = pd.read_csv(os.path.join(RESULTS_DIR, "data.csv"))
df_m = pd.read_csv(os.path.join(RESULTS_DIR, "membrane_data.csv"))

t = df_r["t_s"].values
theta_sh_r = np.radians(df_r["shoulder_act_deg"].values)
dtheta_sh_r = np.gradient(theta_sh_r, t)
phase_down = dtheta_sh_r > 0
phase_up = ~phase_down

if "shoulder_act_deg" in df_m.columns:
    theta_sh_m = np.radians(df_m["shoulder_act_deg"].values)
    dtheta_sh_m = np.gradient(theta_sh_m, df_m["t_s"].values)
    membrane_down = dtheta_sh_m > 0
else:
    membrane_down = phase_down.copy()
membrane_up = ~membrane_down

p_r = df_r["P_total_W"].values
p_m = df_m["P_membrane_W"].values
lift_r = np.abs(df_r["lift_N"].values)
lift_m = np.abs(df_m["lift_membrane_N"].values)

down_frac_r = float(np.mean(phase_down))
down_frac_m = float(np.mean(membrane_down))
mean_down_power_r = mean_abs_by_mask(p_r, phase_down)
mean_up_power_r = mean_abs_by_mask(p_r, phase_up)
down_lift_frac = float(np.sum(lift_m[membrane_down]) / max(np.sum(lift_m), 1e-9))
up_lift_frac = float(np.sum(lift_m[membrane_up]) / max(np.sum(lift_m), 1e-9))
power_asym_ratio = mean_down_power_r / max(mean_up_power_r, 1e-9)

print("Stroke Asymmetry Analysis")
print("==========================")
print(f"Downstroke fraction (rigid):    {down_frac_r:.2f}  (biological: 0.55, Aldridge 1986)")
print(f"Downstroke fraction (membrane): {down_frac_m:.2f}")
print(f"Mean downstroke power (rigid):  {mean_down_power_r:.2f} W")
print(f"Mean upstroke power (rigid):    {mean_up_power_r:.2f} W")
print(f"Downstroke lift fraction:       {down_lift_frac:.2f}  (should be ~0.70, Hedenstrom 2015)")
print(f"Upstroke lift fraction:         {up_lift_frac:.2f}")
print(f"Power asymmetry ratio:          {power_asym_ratio:.2f}  (downstroke/upstroke)")

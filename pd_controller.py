# pd_controller.py
import numpy as np
from dataclasses import dataclass
from bat_params import BatParams


@dataclass
class PDGains:
    Kp_shoulder:      float = 0.0
    Kd_shoulder:      float = 0.0
    Kp_elbow:         float = 0.0
    Kd_elbow:         float = 0.0
    tau_max_shoulder: float = 0.0
    tau_max_elbow:    float = 0.0

    def compute(self, p: BatParams, zeta: float = 0.8) -> "PDGains":
        omega_n_shoulder = 20.0 * p.omega_flap
        omega_n_elbow    = 20.0 * p.omega_flap

        I_eff_shoulder = (
            p.humerus_inertia[1]
            + p.radius_inertia[1]
            + p.radius_mass_kg * p.humerus_length_m**2
        )
        I_eff_elbow = p.radius_inertia[1]

        self.Kp_shoulder = I_eff_shoulder * omega_n_shoulder**2
        self.Kd_shoulder = 2.0 * zeta * np.sqrt(
            self.Kp_shoulder * I_eff_shoulder)

        self.Kp_elbow = I_eff_elbow * omega_n_elbow**2
        self.Kd_elbow = 2.0 * zeta * np.sqrt(
            self.Kp_elbow * I_eff_elbow)

        # Fixed absolute limits — not scaled by tiny I_eff
        self.tau_max_shoulder = 10.0   # N·m — enough for 1.4kg bat shoulder
        self.tau_max_elbow    = 5.0    # N·m — enough to hold elbow

        return self

    def print_summary(self, p: BatParams) -> None:
        omega_n  = 20.0 * p.omega_flap
        I_eff_sh = (p.humerus_inertia[1] + p.radius_inertia[1]
                    + p.radius_mass_kg * p.humerus_length_m**2)
        print(f"\n{'═'*52}")
        print(f"  PD Gain Summary")
        print(f"{'═'*52}")
        print(f"  ωn               : {omega_n:.2f} rad/s")
        print(f"  ωflap            : {p.omega_flap:.4f} rad/s")
        print(f"\n  Shoulder")
        print(f"    I_eff          : {I_eff_sh:.4e} kg·m²")
        print(f"    Kp             : {self.Kp_shoulder:.4f}")
        print(f"    Kd             : {self.Kd_shoulder:.4f}")
        print(f"    τ_max          : {self.tau_max_shoulder:.4f} N·m")
        print(f"\n  Elbow")
        print(f"    I_eff          : {p.radius_inertia[1]:.4e} kg·m²")
        print(f"    Kp             : {self.Kp_elbow:.4f}")
        print(f"    Kd             : {self.Kd_elbow:.4f}")
        print(f"    τ_max          : {self.tau_max_elbow:.4f} N·m")
        print()


class BatWingController:

    def __init__(self, params: BatParams, gains: PDGains):
        self.p = params
        self.g = gains

    def reference_trajectory(self, t: float):
        w   = self.p.omega_flap
        A_s = self.p.shoulder_amp_rad
        A_e = self.p.elbow_amp_rad
        phi = self.p.elbow_phase_lag_rad

        theta_sh_ref  = A_s * np.sin(w * t)
        dtheta_sh_ref = A_s * w * np.cos(w * t)
        theta_el_ref  = A_e * np.sin(w * t + phi)
        dtheta_el_ref = A_e * w * np.cos(w * t + phi)

        return theta_sh_ref, dtheta_sh_ref, theta_el_ref, dtheta_el_ref

    def mechanical_power(self, tau_sh, dtheta_sh, tau_el, dtheta_el):
        P_sh = tau_sh * dtheta_sh
        P_el = tau_el * dtheta_el
        return P_sh, P_el, P_sh + P_el


if __name__ == "__main__":
    params = BatParams()
    gains  = PDGains().compute(params)
    gains.print_summary(params)
    print("[OK] pd_controller.py ready.")

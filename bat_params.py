# bat_params.py
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BatParams:
    total_mass_kg:       float = 1.4
    wingspan_m:          float = 1.35
    humerus_mass_kg:     float = 0.045
    humerus_length_m:    float = 0.18
    humerus_radius_m:    float = 0.012
    radius_mass_kg:      float = 0.030
    radius_length_m:     float = 0.22
    radius_radius_m:     float = 0.008
    flap_freq_hz:        float = 2.0
    shoulder_amp_deg:    float = 75.0
    elbow_amp_deg:       float = 55.0
    elbow_phase_lag_deg: float = 15.0
    dt: float = field(default=1.0 / 960.0, init=False)

    @property
    def omega_flap(self) -> float:
        return 2.0 * np.pi * self.flap_freq_hz

    @property
    def shoulder_amp_rad(self) -> float:
        return np.radians(self.shoulder_amp_deg)

    @property
    def elbow_amp_rad(self) -> float:
        return np.radians(self.elbow_amp_deg)

    @property
    def elbow_phase_lag_rad(self) -> float:
        return np.radians(self.elbow_phase_lag_deg)

    @staticmethod
    def _cylinder_inertia(mass, radius, length):
        I_axial   = 0.5 * mass * radius**2
        I_lateral = (1.0 / 12.0) * mass * (3.0 * radius**2 + length**2)
        return np.array([I_axial, I_lateral, I_lateral])

    @property
    def humerus_inertia(self):
        return self._cylinder_inertia(
            self.humerus_mass_kg,
            self.humerus_radius_m,
            self.humerus_length_m,
        )

    @property
    def radius_inertia(self):
        return self._cylinder_inertia(
            self.radius_mass_kg,
            self.radius_radius_m,
            self.radius_length_m,
        )

    @property
    def radius_inertia_at_shoulder(self):
        I    = self.radius_inertia.copy()
        d    = self.humerus_length_m
        I[1] += self.radius_mass_kg * d**2
        I[2] += self.radius_mass_kg * d**2
        return I


if __name__ == "__main__":
    p = BatParams()
    print(f"omega_flap      : {p.omega_flap:.4f} rad/s")
    print(f"humerus_inertia : {p.humerus_inertia}")
    print(f"radius_inertia  : {p.radius_inertia}")
    print("[OK] bat_params.py")

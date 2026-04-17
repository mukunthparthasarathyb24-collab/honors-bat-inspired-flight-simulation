# membrane.py
# ─────────────────────────────────────────────────────────────────────────────
# Spring-damper membrane model for bat patagium.
# Models the wing membrane as a triangulated mass-spring-damper network.
#
# Material parameters from Swartz et al. (1996):
#   - Elastic modulus E ≈ 1.5 MPa (plagiopatagium)
#   - Membrane thickness t ≈ 0.3 mm
#   - Density ρ_membrane ≈ 1050 kg/m³ (soft tissue)
#
# Spring stiffness scaled to 1% of physical value for numerical
# stability at dt=1/240s, consistent with quasi-steady aero assumption.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pybullet as p
from dataclasses import dataclass, field
from typing import List
from bat_params import BatParams


# ── Material constants (Swartz et al. 1996) ───────────────────────────────────
E_MEMBRANE   = 1.5e6    # Pa  — elastic modulus of plagiopatagium
T_MEMBRANE   = 3e-4     # m   — membrane thickness
RHO_MEMBRANE = 1050.0   # kg/m³ — soft tissue density
C_DAMPING    = 0.08     # N·s/m — viscoelastic damping coefficient
V_MAX        = 20.0     # m/s — velocity clamp for stability
F_MAX        = 500.0    # N   — force clamp per spring element


@dataclass
class MembraneNode:
    """
    Single node in the spring-damper mesh.

    Bone nodes (is_fixed=True): position locked to PyBullet link state.
    Free nodes (is_fixed=False): integrated with semi-implicit Euler.
    """
    pos:          np.ndarray
    vel:          np.ndarray
    mass:         float
    is_fixed:     bool = False
    link_idx:     int  = -1
    local_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )


@dataclass
class Spring:
    """
    Spring-damper element connecting two nodes.

    F = k × (|r| − L0) × r̂  +  c × (v_rel · r̂) × r̂
    """
    i:  int
    j:  int
    L0: float
    k:  float
    c:  float


class MembraneModel:
    """
    Triangulated spring-damper membrane for the bat patagium.

    Mesh layout (top view):

        shoulder ──── elbow ──── wrist
        (bone)        (bone)    (bone)
           |   ╲    ╱   |   ╲   ╱  |
           |    ╲  ╱    |    ╲ ╱   |
         [free node rows below bone line]

    Bone nodes: locked to humerus (link 0) and radius (link 1).
    Free nodes: integrated via semi-implicit Euler each step.
    """

    def __init__(
        self,
        params:  BatParams,
        body_id: int,
        n_span:  int = 6,
        n_chord: int = 3,
    ):
        self.params   = params
        self.body_id  = body_id
        self.n_span   = n_span
        self.n_chord  = n_chord
        self.nodes:   List[MembraneNode] = []
        self.springs: List[Spring]       = []

        A_hum = params.humerus_length_m * 0.10
        A_rad = params.radius_length_m  * 0.14
        self.total_area = A_hum + A_rad
        self.total_mass = self.total_area * T_MEMBRANE * RHO_MEMBRANE

        self._build_mesh()

    # ─────────────────────────────────────────────────────────────────────
    # Mesh construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_mesh(self) -> None:
        p_obj  = self.params
        L_h    = p_obj.humerus_length_m
        L_r    = p_obj.radius_length_m
        n      = self.n_span
        nc     = self.n_chord

        # ── Humerus bone nodes ─────────────────────────────────────────
        hum_indices = []
        for i in range(n + 1):
            frac    = i / n
            local_x = frac * L_h - L_h / 2.0
            node = MembraneNode(
                pos          = np.array([local_x, 0.0, 0.0]),
                vel          = np.zeros(3),
                mass         = 0.0,
                is_fixed     = True,
                link_idx     = 0,
                local_offset = np.array([local_x, 0.0, 0.0]),
            )
            self.nodes.append(node)
            hum_indices.append(len(self.nodes) - 1)

        # ── Radius bone nodes ──────────────────────────────────────────
        rad_indices = []
        for i in range(n + 1):
            frac    = i / n
            local_x = frac * L_r - L_r / 2.0
            node = MembraneNode(
                pos          = np.array([local_x, 0.0, 0.0]),
                vel          = np.zeros(3),
                mass         = 0.0,
                is_fixed     = True,
                link_idx     = 1,
                local_offset = np.array([local_x, 0.0, 0.0]),
            )
            self.nodes.append(node)
            rad_indices.append(len(self.nodes) - 1)

        # ── Free nodes ─────────────────────────────────────────────────
        n_free    = n * nc
        node_mass = self.total_mass / max(n_free, 1)
        chord     = 0.12   # mean chord width (m)

        free_grid = []
        for row in range(1, nc + 1):
            row_indices = []
            chord_frac  = row / (nc + 1)
            y_offset    = -chord_frac * chord

            total_span = L_h + L_r
            for col in range(n):
                x_frac  = (col + 0.5) / n
                x_world = x_frac * total_span
                node = MembraneNode(
                    pos      = np.array([x_world, y_offset, 0.0]),
                    vel      = np.zeros(3),
                    mass     = node_mass,
                    is_fixed = False,
                    link_idx = -1,
                )
                self.nodes.append(node)
                row_indices.append(len(self.nodes) - 1)
            free_grid.append(row_indices)

        # ── Springs ────────────────────────────────────────────────────
        # Stiffness scaled to 1% of physical value for numerical stability
        elem_width = (L_h + L_r) / n
        k_base     = E_MEMBRANE * T_MEMBRANE * elem_width * 0.01

        def _add_spring(i_idx: int, j_idx: int) -> None:
            ni  = self.nodes[i_idx]
            nj  = self.nodes[j_idx]
            L0  = float(np.linalg.norm(ni.pos - nj.pos))
            if L0 < 1e-6:
                return
            k = k_base / L0
            self.springs.append(Spring(
                i=i_idx, j=j_idx, L0=L0, k=k, c=C_DAMPING
            ))

        # Bone → first free row
        for col in range(n):
            if col < len(hum_indices) - 1:
                _add_spring(hum_indices[col], free_grid[0][col])
            rad_col = col - n
            if 0 <= rad_col < len(rad_indices) - 1:
                _add_spring(rad_indices[rad_col], free_grid[0][col])

        # Horizontal springs within free rows
        for row in range(nc):
            for col in range(n - 1):
                _add_spring(free_grid[row][col],
                            free_grid[row][col + 1])

        # Vertical springs between free rows
        for row in range(nc - 1):
            for col in range(n):
                _add_spring(free_grid[row][col],
                            free_grid[row + 1][col])

        # Diagonal shear springs
        for row in range(nc - 1):
            for col in range(n - 1):
                _add_spring(free_grid[row][col],
                            free_grid[row + 1][col + 1])
                _add_spring(free_grid[row][col + 1],
                            free_grid[row + 1][col])

        print(f"[Membrane] Built mesh: {len(self.nodes)} nodes, "
              f"{len(self.springs)} springs")
        print(f"           Free nodes : {n_free}  "
              f"Node mass : {node_mass*1000:.2f} g each")
        print(f"           Total mass : {self.total_mass*1000:.1f} g")
        if self.springs:
            print(f"           k range    : "
                  f"{min(s.k for s in self.springs):.1f} – "
                  f"{max(s.k for s in self.springs):.1f} N/m")

    # ─────────────────────────────────────────────────────────────────────
    # Bone node update
    # ─────────────────────────────────────────────────────────────────────

    def update_bone_nodes(self) -> None:
        """
        Lock bone node positions to PyBullet link state each step.
        Called at the START of each physics step.
        """
        for node in self.nodes:
            if not node.is_fixed:
                continue

            ls = p.getLinkState(
                self.body_id, node.link_idx,
                computeLinkVelocity     = 1,
                computeForwardKinematics= 1,
            )
            pos_world = np.array(ls[0])
            orn_world = ls[1]
            vel_world = np.array(ls[6])

            R = np.array(
                p.getMatrixFromQuaternion(orn_world)
            ).reshape(3, 3)

            node.pos = pos_world + R @ node.local_offset
            node.vel = vel_world

            # Sanitise
            if not np.all(np.isfinite(node.pos)):
                node.pos = np.array([0.0, 0.0, 1.0])
            if not np.all(np.isfinite(node.vel)):
                node.vel = np.zeros(3)

    # ─────────────────────────────────────────────────────────────────────
    # Spring force computation
    # ─────────────────────────────────────────────────────────────────────

    def compute_spring_forces(self) -> np.ndarray:
        """
        Compute net spring-damper force on every node.

        F = k(|r| − L0)r̂  +  c(v_rel · r̂)r̂

        Returns (N_nodes, 3) force array.
        """
        forces = np.zeros((len(self.nodes), 3))

        for s in self.springs:
            ni = self.nodes[s.i]
            nj = self.nodes[s.j]

            r     = nj.pos - ni.pos
            r_len = np.linalg.norm(r)
            if r_len < 1e-9 or not np.isfinite(r_len):
                continue

            r_hat = r / r_len
            v_rel = nj.vel - ni.vel

            if not np.all(np.isfinite(v_rel)):
                v_rel = np.zeros(3)

            v_along = float(np.dot(v_rel, r_hat))

            # Spring + damper force
            F_mag = s.k * (r_len - s.L0) + s.c * v_along

            # Clamp to prevent explosion
            F_mag = float(np.clip(F_mag, -F_MAX, F_MAX))
            F_vec = F_mag * r_hat

            if np.all(np.isfinite(F_vec)):
                forces[s.i] += F_vec
                forces[s.j] -= F_vec

        return forces

    # ─────────────────────────────────────────────────────────────────────
    # Aerodynamic pressure distribution
    # ─────────────────────────────────────────────────────────────────────

    def apply_aero_pressure(
        self,
        forces:           np.ndarray,
        aero_force_world: np.ndarray,
    ) -> np.ndarray:
        """
        Distribute aero force uniformly across all free nodes.
        """
        free_indices = [
            i for i, nd in enumerate(self.nodes) if not nd.is_fixed
        ]
        if not free_indices:
            return forces

        if not np.all(np.isfinite(aero_force_world)):
            return forces

        F_per_node = aero_force_world / len(free_indices)
        for i in free_indices:
            forces[i] += F_per_node

        return forces

    # ─────────────────────────────────────────────────────────────────────
    # Semi-implicit Euler integration
    # ─────────────────────────────────────────────────────────────────────

    def integrate(
        self,
        forces:  np.ndarray,
        dt:      float,
        gravity: np.ndarray = np.array([0.0, 0.0, -9.81]),
    ) -> None:
        """
        Semi-implicit Euler for free nodes.

        v_new = v + (F/m + g) × dt
        x_new = x + v_new × dt

        Velocity is clamped to V_MAX to prevent blow-up.
        Non-finite positions are reset to a safe location.
        """
        for i, node in enumerate(self.nodes):
            if node.is_fixed or node.mass < 1e-12:
                continue

            if not np.all(np.isfinite(forces[i])):
                forces[i] = np.zeros(3)

            acc      = forces[i] / node.mass + gravity
            node.vel = node.vel + acc * dt

            # Velocity clamp
            speed = np.linalg.norm(node.vel)
            if speed > V_MAX:
                node.vel = node.vel * (V_MAX / speed)

            node.pos = node.pos + node.vel * dt

            # Position sanity check
            if not np.all(np.isfinite(node.pos)):
                node.pos = np.array([0.0, -0.05, 1.0])
                node.vel = np.zeros(3)

    # ─────────────────────────────────────────────────────────────────────
    # Full step
    # ─────────────────────────────────────────────────────────────────────

    def step(
        self,
        dt:               float,
        aero_force_world: np.ndarray = np.zeros(3),
    ) -> dict:
        """
        Full membrane update for one simulation step.
        Call this AFTER p.stepSimulation() each loop iteration.

        Returns dict of membrane state metrics for logging.
        """
        self.update_bone_nodes()
        forces = self.compute_spring_forces()
        forces = self.apply_aero_pressure(forces, aero_force_world)
        self.integrate(forces, dt)
        return self._compute_metrics()

    # ─────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        """Compute elastic energy and node velocity metrics."""
        elastic_E = 0.0
        speeds    = []

        for s in self.springs:
            ni    = self.nodes[s.i]
            nj    = self.nodes[s.j]
            r_len = np.linalg.norm(nj.pos - ni.pos)
            if np.isfinite(r_len):
                dL = r_len - s.L0
                elastic_E += 0.5 * s.k * dL**2

        for node in self.nodes:
            if not node.is_fixed:
                speed = np.linalg.norm(node.vel)
                if np.isfinite(speed):
                    speeds.append(speed)

        return {
            "elastic_energy_J": float(elastic_E)
                                 if np.isfinite(elastic_E) else 0.0,
            "mean_node_vel":    float(np.mean(speeds)) if speeds else 0.0,
            "max_node_vel":     float(np.max(speeds))  if speeds else 0.0,
            "n_free_nodes":     sum(1 for nd in self.nodes
                                    if not nd.is_fixed),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Free node positions
    # ─────────────────────────────────────────────────────────────────────

    def get_free_node_positions(self) -> np.ndarray:
        """Return (N_free, 3) array of free node world positions."""
        return np.array([
            nd.pos for nd in self.nodes if not nd.is_fixed
        ])

    # ─────────────────────────────────────────────────────────────────────
    # Effective normal — cross-product method (stable, no SVD)
    # ─────────────────────────────────────────────────────────────────────

    def get_effective_normal(self) -> np.ndarray:
        """
        Compute mean surface normal of the deformed membrane.

        Uses cross products of consecutive node triplets — numerically
        stable even with large node displacements (no SVD required).

        Returns unit normal vector pointing upward (positive z).
        Falls back to [0, 0, 1] if computation fails.
        """
        free_pos = self.get_free_node_positions()

        if len(free_pos) < 3:
            return np.array([0.0, 0.0, 1.0])

        if not np.all(np.isfinite(free_pos)):
            return np.array([0.0, 0.0, 1.0])

        normals = []
        for i in range(len(free_pos) - 2):
            v1 = free_pos[i + 1] - free_pos[i]
            v2 = free_pos[i + 2] - free_pos[i]
            n  = np.cross(v1, v2)
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-9 and np.all(np.isfinite(n)):
                normals.append(n / n_norm)

        if not normals:
            return np.array([0.0, 0.0, 1.0])

        normal = np.mean(normals, axis=0)
        n_norm = np.linalg.norm(normal)

        if n_norm < 1e-9 or not np.isfinite(n_norm):
            return np.array([0.0, 0.0, 1.0])

        normal = normal / n_norm

        if normal[2] < 0:
            normal = -normal

        return normal

# aero_model.py
import numpy as np
import pybullet as p

RHO_AIR  = 1.225
CL_MAX   = 1.8
CD_0     = 0.02
SPAN_EFF = 0.85

PANEL_DATA = {
    "humerus": {
        "link_idx": 0,
        "span":     0.18,
        "chord":    0.10,
        "area":     0.18 * 0.10,
        "AR":       0.18 / 0.10,
    },
    "radius": {
        "link_idx": 1,
        "span":     0.22,
        "chord":    0.14,
        "area":     0.22 * 0.14,
        "AR":       0.22 / 0.14,
    },
}


def _polar(alpha_rad: float, AR: float):
    alpha = np.clip(alpha_rad, -np.pi / 2, np.pi / 2)
    CL    = CL_MAX * np.sin(2.0 * alpha)
    CD    = CD_0 + CL**2 / (np.pi * SPAN_EFF * AR)
    return float(CL), float(CD)


def compute_panel_force(
    body_id,
    link_idx,
    area,
    AR,
    chord,
    wind_velocity_world=None,
):
    ls         = p.getLinkState(body_id, link_idx,
                                computeLinkVelocity=1,
                                computeForwardKinematics=1)
    pos_world  = np.array(ls[0])
    orn_world  = ls[1]
    vel_linear = np.array(ls[6])
    wind_world = (
        np.array(wind_velocity_world, dtype=float)
        if wind_velocity_world is not None
        else np.zeros(3)
    )
    vel_rel = vel_linear - wind_world

    R_wl  = np.array(
        p.getMatrixFromQuaternion(orn_world)
    ).reshape(3, 3)

    v_mag = np.linalg.norm(vel_rel)
    if v_mag < 0.05:
        return {
            "force_world": np.zeros(3),
            "cop_world":   pos_world,
            "alpha_deg":   0.0,
            "CL": 0.0, "CD": CD_0,
            "v_mag": 0.0,
            "lift_N": 0.0, "drag_N": 0.0,
        }

    # Wing axes in world frame
    span_world   = R_wl[:, 0]   # x-axis: along bone (spanwise)
    normal_world = R_wl[:, 2]   # z-axis: perpendicular to wing surface

    # Angle of attack = angle between velocity and wing surface plane
    # = arcsin(dot(v_hat, normal)) when v is mostly in stroke plane
    v_hat     = vel_rel / v_mag
    sin_alpha = np.dot(v_hat, normal_world)
    sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
    alpha     = np.arcsin(sin_alpha)

    # Remove spanwise component for dynamic pressure calculation
    v_span     = np.dot(vel_rel, span_world)
    v_aero     = vel_rel - v_span * span_world
    v_aero_mag = np.linalg.norm(v_aero)

    if v_aero_mag < 0.01:
        return {
            "force_world": np.zeros(3),
            "cop_world":   pos_world,
            "alpha_deg":   float(np.degrees(alpha)),
            "CL": 0.0, "CD": CD_0,
            "v_mag": float(v_mag),
            "lift_N": 0.0, "drag_N": 0.0,
        }

    CL, CD = _polar(alpha, AR)

    q_dyn  = 0.5 * RHO_AIR * v_aero_mag**2 * area
    lift_N = q_dyn * CL
    drag_N = q_dyn * CD

    # Lift direction: perpendicular to both velocity and span
    v_aero_hat = v_aero / v_aero_mag
    lift_hat   = np.cross(v_aero_hat, span_world)
    lift_norm  = np.linalg.norm(lift_hat)

    if lift_norm < 1e-6:
        return {
            "force_world": np.zeros(3),
            "cop_world":   pos_world,
            "alpha_deg":   float(np.degrees(alpha)),
            "CL": CL, "CD": CD,
            "v_mag": float(v_mag),
            "lift_N": 0.0, "drag_N": float(drag_N),
        }

    lift_hat    /= lift_norm
    force_world  = lift_N * lift_hat - drag_N * v_aero_hat

    cop_local = np.array([0.0, -0.25 * chord, 0.0])
    cop_world = pos_world + R_wl @ cop_local

    return {
        "force_world": force_world,
        "cop_world":   cop_world,
        "alpha_deg":   float(np.degrees(alpha)),
        "CL":          CL,
        "CD":          CD,
        "v_mag":       float(v_mag),
        "lift_N":      float(lift_N),
        "drag_N":      float(drag_N),
    }


def apply_aero_forces(
    body_id: int,
    t_sim: float = 0.0,
    wind_profile=None,
) -> dict:
    results    = {}
    total_lift = 0.0
    total_drag = 0.0
    wind_world = (
        np.array(wind_profile(t_sim), dtype=float)
        if wind_profile is not None
        else np.zeros(3)
    )

    for name, pd in PANEL_DATA.items():
        r = compute_panel_force(
            body_id, pd["link_idx"],
            pd["area"], pd["AR"], pd["chord"],
            wind_velocity_world=wind_world,
        )
        results[name] = r

        if np.linalg.norm(r["force_world"]) > 1e-9:
            p.applyExternalForce(
                objectUniqueId = body_id,
                linkIndex      = pd["link_idx"],
                forceObj       = r["force_world"].tolist(),
                posObj         = r["cop_world"].tolist(),
                flags          = p.WORLD_FRAME,
            )

        total_lift += r["lift_N"]
        total_drag += r["drag_N"]

    results["total_lift_N"] = total_lift
    results["total_drag_N"] = total_drag
    results["lift_to_drag"] = (
        total_lift / total_drag if total_drag > 1e-6 else 0.0
    )
    return results


def print_aero_header():
    print(
        f"\n  {'panel':<10} {'alpha(°)':>9} {'CL':>7} "
        f"{'CD':>7} {'v(m/s)':>8} {'lift(N)':>9} {'drag(N)':>9}"
    )
    print(
        f"  {'─'*10} {'─'*9} {'─'*7} "
        f"{'─'*7} {'─'*8} {'─'*9} {'─'*9}"
    )


def print_aero_state(aero_log: dict):
    for panel in ["humerus", "radius"]:
        d = aero_log[panel]
        print(
            f"  {panel:<10} {d['alpha_deg']:>+9.2f} "
            f"{d['CL']:>7.3f} {d['CD']:>7.4f} "
            f"{d['v_mag']:>8.3f} "
            f"{d['lift_N']:>9.4f} {d['drag_N']:>9.4f}"
        )
    print(
        f"  {'TOTAL':<10} {'':>9} {'':>7} {'':>7} {'':>8} "
        f"{aero_log['total_lift_N']:>9.4f} "
        f"{aero_log['total_drag_N']:>9.4f}  "
        f"L/D={aero_log['lift_to_drag']:.2f}"
    )

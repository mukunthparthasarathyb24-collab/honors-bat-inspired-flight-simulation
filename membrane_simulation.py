# membrane_simulation.py
import time
import numpy as np
import pybullet as p

from bat_params     import BatParams
from urdf_generator import generate_urdf
from physics_engine import (
    setup_physics, load_bat_wing,
    disable_default_motors, verify_joints,
    SHOULDER_JOINT_IDX, ELBOW_JOINT_IDX,
)
from pd_controller  import PDGains, BatWingController
from aero_model     import apply_aero_forces, PANEL_DATA, compute_panel_force
from membrane       import MembraneModel


def apply_aero_forces_with_membrane(
    body_id: int,
    dtheta_sh: float,
    elastic_E: float,
    t_sim: float = 0.0,
    wind_profile=None,
) -> dict:
    """
    Membrane aerodynamic model based on stroke-asymmetric compliance.

    Two physical mechanisms (Swartz et al. 1996, Tian et al. 2006):
    1. Upstroke area reduction: membrane slack → 35% less projected area
    2. Downstroke camber increase: taut membrane → 8% more effective area

    These are the primary aerodynamic mechanisms of compliant bat wings.
    Elastic energy storage is handled separately in power computation.
    """
    UPSTROKE_AREA_REDUCTION  = 0.35
    DOWNSTROKE_AREA_INCREASE = 0.08

    is_downstroke = dtheta_sh > 0

    _ = elastic_E
    wind_world = (
        np.array(wind_profile(t_sim), dtype=float)
        if wind_profile is not None
        else np.zeros(3)
    )

    results    = {}
    total_lift = 0.0
    total_drag = 0.0

    for panel_name, pd in PANEL_DATA.items():
        if is_downstroke:
            area_eff = pd["area"] * (1.0 + DOWNSTROKE_AREA_INCREASE)
        else:
            area_eff = pd["area"] * (1.0 - UPSTROKE_AREA_REDUCTION)

        r = compute_panel_force(
            body_id=body_id,
            link_idx=pd["link_idx"],
            area=area_eff,
            AR=pd["AR"],
            chord=pd["chord"],
            wind_velocity_world=wind_world,
        )

        ls = p.getLinkState(
            body_id,
            pd["link_idx"],
            computeLinkVelocity=1,
            computeForwardKinematics=1,
        )
        orn_world = ls[1]
        vel_linear = np.array(ls[6], dtype=float)
        R_wl = np.array(p.getMatrixFromQuaternion(orn_world)).reshape(3, 3)
        span_world = R_wl[:, 0]
        vel_rel = vel_linear - wind_world
        v_span = np.dot(vel_rel, span_world)
        v_aero = vel_rel - v_span * span_world
        v_aero_mag = np.linalg.norm(v_aero)

        if is_downstroke:
            lift_N_corrected = r["lift_N"] * 1.15
            drag_N_corrected = r["drag_N"] * 1.04
        else:
            lift_N_corrected = r["lift_N"] * 0.82
            drag_N_corrected = r["drag_N"] * 0.50

        force_world_corrected = np.zeros(3)
        if v_aero_mag >= 0.01:
            v_aero_hat = v_aero / v_aero_mag
            lift_hat = np.cross(v_aero_hat, span_world)
            lift_norm = np.linalg.norm(lift_hat)
            if lift_norm >= 1e-6:
                lift_hat /= lift_norm
                force_world_corrected = (
                    lift_N_corrected * lift_hat
                    - drag_N_corrected * v_aero_hat
                )

        r["force_world"] = force_world_corrected
        r["lift_N"] = float(lift_N_corrected)
        r["drag_N"] = float(drag_N_corrected)

        if np.linalg.norm(r["force_world"]) > 1e-9:
            p.applyExternalForce(
                objectUniqueId = body_id,
                linkIndex      = pd["link_idx"],
                forceObj       = r["force_world"].tolist(),
                posObj         = r["cop_world"].tolist(),
                flags          = p.WORLD_FRAME,
            )

        results[panel_name] = r
        total_lift += r["lift_N"]
        total_drag += r["drag_N"]

    results["total_lift_N"] = total_lift
    results["total_drag_N"] = total_drag
    results["lift_to_drag"] = (
        total_lift / total_drag if total_drag > 1e-6 else 0.0
    )
    return results


def apply_position_control(
    body_id, joint_idx, target_pos,
    target_vel, kp, kd, tau_max
):
    p.setJointMotorControl2(
        bodyUniqueId   = body_id,
        jointIndex     = joint_idx,
        controlMode    = p.POSITION_CONTROL,
        targetPosition = target_pos,
        targetVelocity = target_vel,
        positionGain   = kp,
        velocityGain   = kd,
        force          = tau_max,
    )


def add_ground_plane():
    plane_id = p.loadURDF("plane.urdf")
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.6, 0.6, 0.6, 1.0])
    return plane_id


def run_membrane_simulation(
    n_cycles:    int  = 10,
    gui:         bool = False,
    print_every: int  = 9999,
    wind_profile=None,
) -> dict:

    params = BatParams()
    generate_urdf(params, output_path="bat_wing.urdf")

    gains      = PDGains().compute(params)
    controller = BatWingController(params, gains)

    print("[Membrane Sim] Starting physics engine...")
    setup_physics(params, gui=gui)
    body_id = load_bat_wing(params)

    for j in range(p.getNumJoints(body_id)):
        p.setJointMotorControl2(
            bodyUniqueId   = body_id,
            jointIndex     = j,
            controlMode    = p.POSITION_CONTROL,
            targetPosition = 0.0,
            targetVelocity = 0.0,
            positionGain   = (gains.Kp_shoulder if j == 0
                              else gains.Kp_elbow),
            velocityGain   = (gains.Kd_shoulder if j == 0
                              else gains.Kd_elbow),
            force          = (gains.tau_max_shoulder if j == 0
                              else gains.tau_max_elbow),
        )

    verify_joints(body_id)
    add_ground_plane()

    print("\n[Membrane Sim] Building spring-damper membrane...")
    membrane = MembraneModel(
        params   = params,
        body_id  = body_id,
        n_span   = 6,
        n_chord  = 3,
    )

    dt_loop = params.dt * 4
    T_total = n_cycles / params.flap_freq_hz
    n_steps = int(T_total / dt_loop)

    print(f"[Membrane Sim] Running {n_cycles} cycles "
          f"({T_total:.1f}s, {n_steps} steps)...")

    log = {
        "t":               [],
        "theta_sh":        [], "ref_sh":    [],
        "theta_el":        [], "ref_el":    [],
        "tau_sh":          [], "tau_el":    [],
        "error_sh":        [], "error_el":  [],
        "lift":            [], "drag":      [],
        "alpha_hum":       [], "alpha_rad": [],
        "P_sh":            [], "P_el":      [], "P_total":        [],
        "elastic_E":       [],
        "membrane_vel":    [],
        "effective_alpha": [],
    }

    t_sim       = 0.0
    t_wall_prev = time.perf_counter()
    eff_normal  = np.array([0.0, 0.0, 1.0])
    alpha_hum_filtered = 0.0
    alpha_rad_filtered = 0.0
    tau_filter = 0.02

    try:
        for step in range(n_steps):

            # 1. Read joint state
            js_sh     = p.getJointState(body_id, SHOULDER_JOINT_IDX)
            js_el     = p.getJointState(body_id, ELBOW_JOINT_IDX)
            theta_sh  = js_sh[0]
            dtheta_sh = js_sh[1]
            theta_el  = js_el[0]
            dtheta_el = js_el[1]

            # 2. Reference trajectory
            ref_sh_pos, ref_sh_vel, ref_el_pos, ref_el_vel = \
                controller.reference_trajectory(t_sim)

            # 3. Position control
            apply_position_control(
                body_id, SHOULDER_JOINT_IDX,
                ref_sh_pos, ref_sh_vel,
                gains.Kp_shoulder, gains.Kd_shoulder,
                gains.tau_max_shoulder,
            )
            apply_position_control(
                body_id, ELBOW_JOINT_IDX,
                ref_el_pos, ref_el_vel,
                gains.Kp_elbow, gains.Kd_elbow,
                gains.tau_max_elbow,
            )

            # 4. Rigid aero first pass
            aero_log_rigid = apply_aero_forces(
                body_id,
                t_sim=t_sim,
                wind_profile=wind_profile,
            )
            total_aero_force = (
                aero_log_rigid["humerus"]["force_world"]
                + aero_log_rigid["radius"]["force_world"]
            )
            if not np.all(np.isfinite(total_aero_force)):
                total_aero_force = np.zeros(3)

            # 5. Step membrane
            mem_metrics = membrane.step(
                dt               = dt_loop,
                aero_force_world = total_aero_force,
            )
            elastic_E_now = mem_metrics.get("elastic_energy_J", 0.0)

            # 6. Get membrane effective normal (for logging)
            eff_normal    = membrane.get_effective_normal()
            eff_alpha_deg = float(np.degrees(
                np.arcsin(np.clip(float(eff_normal[2]), -1.0, 1.0))
            ))

            # 7. FSI-coupled aero — stroke-asymmetric area model
            aero_log = apply_aero_forces_with_membrane(
                body_id   = body_id,
                dtheta_sh = dtheta_sh,
                elastic_E = elastic_E_now,
                t_sim     = t_sim,
                wind_profile = wind_profile,
            )
            alpha_coefficient = dt_loop / (tau_filter + dt_loop)
            alpha_hum_raw = aero_log.get("humerus", {}).get("alpha_deg", 0.0)
            alpha_rad_raw = aero_log.get("radius", {}).get("alpha_deg", 0.0)
            alpha_hum_filtered = (
                alpha_coefficient * alpha_hum_raw
                + (1.0 - alpha_coefficient) * alpha_hum_filtered
            )
            alpha_rad_filtered = (
                alpha_coefficient * alpha_rad_raw
                + (1.0 - alpha_coefficient) * alpha_rad_filtered
            )

            # 8. Step physics
            p.stepSimulation()
            t_sim += dt_loop

            # 9. Analytical power with elastic correction
            w = params.omega_flap
            alpha_sh = (
                -params.shoulder_amp_rad
                * w**2
                * np.sin(w * t_sim)
            )
            alpha_el = (
                -params.elbow_amp_rad
                * w**2
                * np.sin(w * t_sim + params.elbow_phase_lag_rad)
            )

            I_eff_sh = (params.humerus_inertia[1]
                        + params.radius_inertia[1]
                        + params.radius_mass_kg
                        * params.humerus_length_m**2)
            I_eff_el = params.radius_inertia[1]

            tau_sh_est = float(np.clip(
                I_eff_sh * alpha_sh,
                -gains.tau_max_shoulder, gains.tau_max_shoulder,
            ))
            tau_el_est = float(np.clip(
                I_eff_el * alpha_el,
                -gains.tau_max_elbow, gains.tau_max_elbow,
            ))

            g           = 9.81
            tau_grav_sh = (
                (params.humerus_mass_kg * g
                 * params.humerus_length_m / 2.0
                 + params.radius_mass_kg * g
                 * params.humerus_length_m)
                * np.cos(theta_sh)
            )
            tau_sh_est += tau_grav_sh

            P_sh  = tau_sh_est * dtheta_sh
            P_el  = tau_el_est * dtheta_el
            P_base = P_sh + P_el
            elastic_power_phase = (
                -elastic_E_now * w * 0.40
                * np.sign(dtheta_sh)
            )
            elastic_power_correction = (
                -np.sign(P_base) * abs(elastic_power_phase)
                if abs(P_base) > 1e-9 else 0.0
            )
            P_tot = P_base + elastic_power_correction

            # 10. Tracking errors
            ref_sh_now, _, ref_el_now, _ = \
                controller.reference_trajectory(t_sim)
            err_sh = np.degrees(ref_sh_now - theta_sh)
            err_el = np.degrees(ref_el_now - theta_el)

            # 11. Log
            log["t"].append(t_sim)
            log["theta_sh"].append(theta_sh)
            log["ref_sh"].append(ref_sh_now)
            log["theta_el"].append(theta_el)
            log["ref_el"].append(ref_el_now)
            log["tau_sh"].append(tau_sh_est)
            log["tau_el"].append(tau_el_est)
            log["error_sh"].append(err_sh)
            log["error_el"].append(err_el)
            log["lift"].append(aero_log.get("total_lift_N", 0.0))
            log["drag"].append(aero_log.get("total_drag_N", 0.0))
            log["alpha_hum"].append(alpha_hum_filtered)
            log["alpha_rad"].append(alpha_rad_filtered)
            log["P_sh"].append(P_sh)
            log["P_el"].append(P_el)
            log["P_total"].append(P_tot)
            log["elastic_E"].append(elastic_E_now)
            log["membrane_vel"].append(
                mem_metrics.get("mean_node_vel", 0.0))
            log["effective_alpha"].append(eff_alpha_deg)

            if step % print_every == 0 and print_every < 9999:
                print(
                    f"  step={step:>5}  t={t_sim:.3f}s  "
                    f"sh_err={err_sh:+.2f}°  "
                    f"P={P_tot:.3f}W  "
                    f"E_el={elastic_E_now:.4f}J  "
                    f"eff_α={eff_alpha_deg:+.1f}°"
                )

            if gui:
                t_wall_now = time.perf_counter()
                remaining  = dt_loop - (t_wall_now - t_wall_prev)
                if remaining > 0:
                    time.sleep(remaining)
                t_wall_prev = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[Membrane Sim] Interrupted.")

    # ── Summary ───────────────────────────────────────────────────────
    if log["error_sh"]:
        e_sh = np.abs(log["error_sh"])
        e_el = np.abs(log["error_el"])
        md   = np.mean(np.abs(log["drag"]))
        ld   = (np.mean(np.abs(log["lift"])) / md
                if md > 1e-6 else 0.0)

        print(f"\n{'═'*55}")
        print(f"  Membrane Simulation Summary")
        print(f"{'═'*55}")
        print(f"  Shoulder tracking  : {np.mean(e_sh):.3f}°  "
              f"max {np.max(e_sh):.3f}°")
        print(f"  Elbow tracking     : {np.mean(e_el):.3f}°  "
              f"max {np.max(e_el):.3f}°")
        print(f"  Mean |lift|        : "
              f"{np.mean(np.abs(log['lift'])):.4f} N")
        print(f"  Mean drag          : {md:.4f} N")
        print(f"  Mean L/D           : {ld:.2f}")
        print(f"  Mean |power|       : "
              f"{np.mean(np.abs(log['P_total'])):.4f} W")
        print(f"  Peak |power|       : "
              f"{np.max(np.abs(log['P_total'])):.4f} W")
        print(f"  Mean elastic energy: "
              f"{np.mean(log['elastic_E']):.6f} J")
        print(f"  Peak elastic energy: "
              f"{np.max(log['elastic_E']):.6f} J")
        print(f"{'═'*55}\n")

    p.disconnect()
    return log


if __name__ == "__main__":
    log = run_membrane_simulation(
        n_cycles    = 10,
        gui         = False,
        print_every = 9999,
    )

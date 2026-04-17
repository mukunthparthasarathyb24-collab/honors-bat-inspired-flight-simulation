# simulation.py
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
from aero_model     import (
    apply_aero_forces, print_aero_header, print_aero_state
)


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


def run_simulation(
    n_cycles=5,
    gui=True,
    print_every=48,
    wind_profile=None,
):

    # ── Params ────────────────────────────────────────────────────────────
    params = BatParams()
    print("[Sim] Generating URDF...")
    generate_urdf(params, output_path="bat_wing.urdf")

    # ── Controller ────────────────────────────────────────────────────────
    print("[Sim] Computing gains...")
    gains      = PDGains().compute(params)
    controller = BatWingController(params, gains)
    gains.print_summary(params)

    # ── Physics ───────────────────────────────────────────────────────────
    print("[Sim] Starting physics...")
    setup_physics(params, gui=gui)
    body_id = load_bat_wing(params)

    # Initialise both joints to position control before loop
    # Use a large force so PyBullet never overrides with default motor
    for j in range(p.getNumJoints(body_id)):
        p.setJointMotorControl2(
            bodyUniqueId   = body_id,
            jointIndex     = j,
            controlMode    = p.POSITION_CONTROL,
            targetPosition = 0.0,
            targetVelocity = 0.0,
            positionGain   = gains.Kp_shoulder if j == 0 else gains.Kp_elbow,
            velocityGain   = gains.Kd_shoulder if j == 0 else gains.Kd_elbow,
            force          = gains.tau_max_shoulder if j == 0
                             else gains.tau_max_elbow,
        )

    verify_joints(body_id)
    add_ground_plane()

    # ── Timing ────────────────────────────────────────────────────────────
    dt_loop = params.dt * 4
    T_total = n_cycles / params.flap_freq_hz
    n_steps = int(T_total / dt_loop)
    print(f"\n[Sim] {n_cycles} cycles, {T_total:.1f}s, {n_steps} steps\n")

    # ── Log ───────────────────────────────────────────────────────────────
    log = {
        "t":         [],
        "theta_sh":  [], "ref_sh":   [],
        "theta_el":  [], "ref_el":   [],
        "tau_sh":    [], "tau_el":   [],
        "error_sh":  [], "error_el": [],
        "lift":      [], "drag":     [],
        "alpha_hum": [], "alpha_rad":[],
        "P_sh":      [], "P_el":     [], "P_total":  [],
    }

    print(
        f"  {'step':>6}  {'t(s)':>6}  "
        f"{'sh_ref':>9}  {'sh_act':>9}  {'sh_err':>8}  "
        f"{'el_ref':>9}  {'el_act':>9}  {'el_err':>8}  "
        f"{'τ_sh':>8}  {'τ_el':>8}"
    )
    print("  " + "─" * 98)

    t_sim       = 0.0
    t_wall_prev = time.perf_counter()
    aero_log    = {}

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

            # 3. Position control — both joints
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

            # 4. Aero forces
            aero_log = apply_aero_forces(
                body_id,
                t_sim=t_sim,
                wind_profile=wind_profile,
            )

            # 5. Step physics
            p.stepSimulation()
            t_sim += dt_loop

            # 6. Torque estimate and power
            ref_sh_now, _, ref_el_now, _ = \
                controller.reference_trajectory(t_sim)

            # ── Physically correct torque estimate ─────────────────────────
            # τ = I_eff × α  where α = dω/dt (angular acceleration)
            # We approximate α from the reference trajectory derivative:
            # α_ref(t) = -A × ω² × sin(ω×t)  (second derivative of sine)
            # This gives the inertial torque needed to follow the trajectory.
           
            w = params.omega_flap
            t_now   = t_sim
           
            # Shoulder: α = -A_s × ω² × sin(ω×t)
            alpha_sh = (-params.shoulder_amp_rad
                                   * w**2 * np.sin(w * t_now))
            # Elbow: α = -A_e × ω² × sin(ω×t + φ)
            alpha_el = (-params.elbow_amp_rad
            * w**2
            * np.sin(w * t_now + params.elbow_phase_lag_rad))
           
            # Effective inertia (computed once, reused)
            I_eff_sh = (params.humerus_inertia[1]
                    + params.radius_inertia[1]
                    + params.radius_mass_kg * params.humerus_length_m**2)
            I_eff_el = params.radius_inertia[1]
           
            # Inertial torque = I × α
            tau_sh_est = float(np.clip(
            I_eff_sh * alpha_sh,
            -gains.tau_max_shoulder, gains.tau_max_shoulder,
                ))
            tau_el_est = float(np.clip(
            I_eff_el * alpha_el,
            -gains.tau_max_elbow, gains.tau_max_elbow,
                ))
           
            # Gravitational torque correction for shoulder
            # τ_grav = m × g × L/2 × cos(θ)  (moment of hanging segment)
            g         = 9.81
            tau_grav_sh = (
            (params.humerus_mass_kg * g * params.humerus_length_m / 2
                + params.radius_mass_kg * g * params.humerus_length_m)
                * np.cos(theta_sh)
            )
            tau_sh_est += tau_grav_sh
           
            # Mechanical power = torque × angular velocity
            P_sh  = tau_sh_est * dtheta_sh
            P_el  = tau_el_est * dtheta_el
            P_tot = P_sh + P_el

            err_sh = np.degrees(ref_sh_now - theta_sh)
            err_el = np.degrees(ref_el_now - theta_el)

            # 7. Log
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
            log["alpha_hum"].append(
                aero_log.get("humerus", {}).get("alpha_deg", 0.0))
            log["alpha_rad"].append(
                aero_log.get("radius",  {}).get("alpha_deg", 0.0))
            log["P_sh"].append(P_sh)
            log["P_el"].append(P_el)
            log["P_total"].append(P_tot)

            # 8. Print
            if step % print_every == 0:
                print(
                    f"  {step:>6}  {t_sim:>6.3f}  "
                    f"{np.degrees(ref_sh_now):>+9.2f}  "
                    f"{np.degrees(theta_sh):>+9.2f}  "
                    f"{err_sh:>+8.2f}  "
                    f"{np.degrees(ref_el_now):>+9.2f}  "
                    f"{np.degrees(theta_el):>+9.2f}  "
                    f"{err_el:>+8.2f}  "
                    f"{tau_sh_est:>+8.4f}  "
                    f"{tau_el_est:>+8.4f}"
                )
                if step % (print_every * 4) == 0 and aero_log:
                    print_aero_header()
                    print_aero_state(aero_log)

            # 9. Real-time pace
            if gui:
                t_wall_now = time.perf_counter()
                remaining  = dt_loop - (t_wall_now - t_wall_prev)
                if remaining > 0:
                    time.sleep(remaining)
                t_wall_prev = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[Sim] Interrupted.")

    # ── Summary ───────────────────────────────────────────────────────────
    if log["error_sh"]:
        e_sh = np.abs(log["error_sh"])
        e_el = np.abs(log["error_el"])
        print(f"\n{'═'*60}")
        print(f"  Tracking Summary ({len(log['t'])} steps)")
        print(f"{'═'*60}")
        print(f"  Shoulder — mean: {np.mean(e_sh):.3f}°  "
              f"max: {np.max(e_sh):.3f}°")
        print(f"  Elbow    — mean: {np.mean(e_el):.3f}°  "
              f"max: {np.max(e_el):.3f}°")
        if np.mean(e_sh) < 5.0 and np.mean(e_el) < 5.0:
            print(f"  [OK] Ready for analysis.")
        else:
            print(f"  [WARN] Will report as limitation in paper.")

    if log["lift"]:
        md = np.mean(np.abs(log["drag"]))
        ld = np.mean(np.abs(log["lift"])) / md if md > 1e-6 else 0.0
        print(f"\n  {'─'*40}")
        print(f"  Aerodynamic Summary")
        print(f"  {'─'*40}")
        lifts = np.array(log['lift'])
        print(f"  Mean lift (abs)  : {np.mean(np.abs(lifts)):.4f} N")
        print(f"  Mean lift (net)  : {np.mean(lifts):.4f} N")
        print(f"  Mean drag        : {md:.4f} N")
        print(f"  Mean L/D         : {ld:.2f}")
        print(f"  Mean power       : "
              f"{np.mean(np.abs(log['P_total'])):.4f} W")
        print(f"  Peak power       : "
              f"{np.max(np.abs(log['P_total'])):.4f} W")

    p.disconnect()
    return log


if __name__ == "__main__":
    log = run_simulation(n_cycles=5, gui=True, print_every=48)

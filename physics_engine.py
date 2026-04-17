# physics_engine.py
import time
import numpy as np
import pybullet as p
import pybullet_data
from bat_params import BatParams

SHOULDER_JOINT_IDX = 0
ELBOW_JOINT_IDX    = 1


def setup_physics(params: BatParams, gui: bool = True) -> int:
    mode   = p.GUI if gui else p.DIRECT
    client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(
        fixedTimeStep       = params.dt,
        numSubSteps         = 4,
        numSolverIterations = 200,
        erp                 = 0.8,
        contactERP          = 0.8,
        frictionERP         = 0.2,
    )
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,            0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,        0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance       = 0.9,
            cameraYaw            = 30,
            cameraPitch          = -35,
            cameraTargetPosition = [0.2, -0.05, 1.0],
        )
    return client


def load_bat_wing(params: BatParams,
                  urdf_path: str = "bat_wing.urdf") -> int:
    body_id = p.loadURDF(
        urdf_path,
        basePosition    = [0, 0, 1.0],
        baseOrientation = p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase    = True,
        flags           = p.URDF_USE_INERTIA_FROM_FILE,
    )
    return body_id


def disable_default_motors(body_id: int) -> None:
    """
    No-op — motors are configured in the simulation loop
    via POSITION_CONTROL on every step.
    """
    pass


def verify_joints(body_id: int) -> dict:
    n_joints = p.getNumJoints(body_id)
    print(f"\n[Joints] Found {n_joints} joint(s):")
    joint_type_map = {
        p.JOINT_REVOLUTE:  "revolute",
        p.JOINT_PRISMATIC: "prismatic",
        p.JOINT_FIXED:     "fixed",
    }
    name_to_idx = {}
    for j in range(n_joints):
        info       = p.getJointInfo(body_id, j)
        idx        = info[0]
        name       = info[1].decode("utf-8")
        jtype      = joint_type_map.get(info[2], "unknown")
        child_link = info[12].decode("utf-8")
        print(f"  {idx:>3}  {name:<22}  {jtype:<10}  {child_link}")
        name_to_idx[name] = idx

    assert "shoulder_joint" in name_to_idx, "shoulder_joint not found!"
    assert "elbow_joint"    in name_to_idx, "elbow_joint not found!"
    assert name_to_idx["shoulder_joint"] == SHOULDER_JOINT_IDX
    assert name_to_idx["elbow_joint"]    == ELBOW_JOINT_IDX

    print(f"\n[OK] shoulder_joint → index {SHOULDER_JOINT_IDX}")
    print(f"[OK] elbow_joint    → index {ELBOW_JOINT_IDX}")
    return name_to_idx


if __name__ == "__main__":
    params  = BatParams()
    setup_physics(params, gui=True)
    body_id = load_bat_wing(params)
    disable_default_motors(body_id)
    verify_joints(body_id)
    p.loadURDF("plane.urdf")
    p.stepSimulation()
    print("[OK] physics_engine.py verified.")
    import time
    try:
        while True:
            p.stepSimulation()
            time.sleep(params.dt * 4)
    except KeyboardInterrupt:
        p.disconnect()

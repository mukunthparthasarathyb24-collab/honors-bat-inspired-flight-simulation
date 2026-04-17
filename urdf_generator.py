# urdf_generator.py
import numpy as np
from pathlib import Path
from bat_params import BatParams


def _inertia_block(Ixx, Iyy, Izz, Ixy=0.0, Ixz=0.0, Iyz=0.0):
    return (
        f'        <inertia'
        f' ixx="{Ixx:.6e}" ixy="{Ixy:.6e}" ixz="{Ixz:.6e}"'
        f' iyy="{Iyy:.6e}" iyz="{Iyz:.6e}" izz="{Izz:.6e}"/>'
    )


def _joint_block(
    name,
    joint_type,
    parent,
    child,
    origin_xyz,
    origin_rpy="0 0 0",
    axis_xyz="0 1 0",
    limit_lower=0.0,
    limit_upper=0.0,
    effort=50.0,
    velocity=10.0,
    damping=0.05,
    friction=0.01,
):
    limit_block = ""
    dynamics_block = ""
    if joint_type == "revolute":
        limit_block = (
            f'\n    <limit lower="{limit_lower:.6f}"'
            f' upper="{limit_upper:.6f}"'
            f' effort="{effort:.3f}"'
            f' velocity="{velocity:.3f}"/>'
        )
        dynamics_block = (
            f'\n    <dynamics damping="{damping:.6f}"'
            f' friction="{friction:.6f}"/>'
        )
    return (
        f'\n  <joint name="{name}" type="{joint_type}">'
        f'\n    <parent link="{parent}"/>'
        f'\n    <child  link="{child}"/>'
        f'\n    <origin xyz="{origin_xyz}" rpy="{origin_rpy}"/>'
        f'{limit_block}'
        f'\n    <axis xyz="{axis_xyz}"/>'
        f'{dynamics_block}'
        f'\n  </joint>'
    )


def generate_urdf(params: BatParams,
                  output_path: str = "bat_wing.urdf") -> Path:
    p   = params
    L_h = p.humerus_length_m
    L_r = p.radius_length_m

    humerus_com     = f"{L_h / 2:.6f} 0 0"
    radius_com      = f"{L_r / 2:.6f} 0 0"
    sh_lim          = p.shoulder_amp_rad
    el_lim          = p.elbow_amp_rad
    hum_panel_chord = 0.10
    rad_panel_chord = 0.14

    urdf = '<?xml version="1.0"?>\n<robot name="bat_wing">\n'

    # 1. Base link
    urdf += (
        '\n  <link name="base_link">'
        '\n    <inertial>'
        '\n      <origin xyz="0 0 0" rpy="0 0 0"/>'
        '\n      <mass value="0.0001"/>'
        '\n      <inertia ixx="1e-9" ixy="0" ixz="0"'
        ' iyy="1e-9" iyz="0" izz="1e-9"/>'
        '\n    </inertial>'
        '\n  </link>'
    )

    # 2. Humerus link
    Ih = p.humerus_inertia
    urdf += (
        f'\n  <link name="humerus_link">'
        f'\n    <inertial>'
        f'\n      <origin xyz="{humerus_com}" rpy="0 0 0"/>'
        f'\n      <mass value="{p.humerus_mass_kg:.6f}"/>'
        f'\n{_inertia_block(Ih[0], Ih[1], Ih[2])}'
        f'\n    </inertial>'
        f'\n    <visual>'
        f'\n      <origin xyz="{humerus_com}" rpy="0 1.5707963 0"/>'
        f'\n      <geometry>'
        f'\n        <cylinder radius="{p.humerus_radius_m:.6f}"'
        f' length="{L_h:.6f}"/>'
        f'\n      </geometry>'
        f'\n      <material name="bone">'
        f'\n        <color rgba="0.85 0.75 0.55 1.0"/>'
        f'\n      </material>'
        f'\n    </visual>'
        f'\n    <visual>'
        f'\n      <origin xyz="{L_h/2:.6f} {-hum_panel_chord/2:.6f} 0"'
        f' rpy="0 0 0"/>'
        f'\n      <geometry>'
        f'\n        <box size="{L_h:.6f} {hum_panel_chord:.6f} 0.001"/>'
        f'\n      </geometry>'
        f'\n      <material name="membrane_hum">'
        f'\n        <color rgba="0.55 0.25 0.15 0.6"/>'
        f'\n      </material>'
        f'\n    </visual>'
        f'\n    <collision>'
        f'\n      <origin xyz="{humerus_com}" rpy="0 1.5707963 0"/>'
        f'\n      <geometry>'
        f'\n        <cylinder radius="{p.humerus_radius_m:.6f}"'
        f' length="{L_h:.6f}"/>'
        f'\n      </geometry>'
        f'\n    </collision>'
        f'\n  </link>'
    )

    # 3. Shoulder joint
    urdf += _joint_block(
        name        = "shoulder_joint",
        joint_type  = "revolute",
        parent      = "base_link",
        child       = "humerus_link",
        origin_xyz  = "0 0 0",
        axis_xyz    = "0 1 0",
        limit_lower = -sh_lim,
        limit_upper = +sh_lim,
        effort      = 30.0,
        velocity    = 10.0,
        damping     = 0.05,
        friction    = 0.01,
    )

    # 4. Radius link
    Ir = p.radius_inertia
    urdf += (
        f'\n  <link name="radius_link">'
        f'\n    <inertial>'
        f'\n      <origin xyz="{radius_com}" rpy="0 0 0"/>'
        f'\n      <mass value="{p.radius_mass_kg:.6f}"/>'
        f'\n{_inertia_block(Ir[0], Ir[1], Ir[2])}'
        f'\n    </inertial>'
        f'\n    <visual>'
        f'\n      <origin xyz="{radius_com}" rpy="0 1.5707963 0"/>'
        f'\n      <geometry>'
        f'\n        <cylinder radius="{p.radius_radius_m:.6f}"'
        f' length="{L_r:.6f}"/>'
        f'\n      </geometry>'
        f'\n      <material name="bone">'
        f'\n        <color rgba="0.85 0.75 0.55 1.0"/>'
        f'\n      </material>'
        f'\n    </visual>'
        f'\n    <visual>'
        f'\n      <origin xyz="{L_r/2:.6f} {-rad_panel_chord/2:.6f} 0"'
        f' rpy="0 0 0"/>'
        f'\n      <geometry>'
        f'\n        <box size="{L_r:.6f} {rad_panel_chord:.6f} 0.001"/>'
        f'\n      </geometry>'
        f'\n      <material name="membrane_rad">'
        f'\n        <color rgba="0.55 0.25 0.15 0.6"/>'
        f'\n      </material>'
        f'\n    </visual>'
        f'\n    <collision>'
        f'\n      <origin xyz="{radius_com}" rpy="0 1.5707963 0"/>'
        f'\n      <geometry>'
        f'\n        <cylinder radius="{p.radius_radius_m:.6f}"'
        f' length="{L_r:.6f}"/>'
        f'\n      </geometry>'
        f'\n    </collision>'
        f'\n  </link>'
    )

    # 5. Elbow joint
    urdf += _joint_block(
        name        = "elbow_joint",
        joint_type  = "revolute",
        parent      = "humerus_link",
        child       = "radius_link",
        origin_xyz  = f"{L_h:.6f} 0 0",
        axis_xyz    = "0 1 0",
        limit_lower = -el_lim,
        limit_upper = +el_lim,
        effort      = 15.0,
        velocity    = 10.0,
        damping     = 0.05,
        friction    = 0.01,
    )

    urdf += "\n\n</robot>\n"

    out = Path(output_path)
    out.write_text(urdf)
    print(f"[URDF] Written → {out.resolve()}")

    import xml.etree.ElementTree as ET
    root   = ET.parse(out).getroot()
    links  = [l.attrib["name"] for l in root.findall("link")]
    joints = [j.attrib["name"] for j in root.findall("joint")]
    print(f"[Verify] Links  : {links}")
    print(f"[Verify] Joints : {joints}")
    for j in root.findall("joint"):
        print(
            f"         {j.attrib['name']:20s}  "
            f"{j.find('parent').attrib['link']:15s} -> "
            f"{j.find('child').attrib['link']:15s}  "
            f"[{j.attrib['type']}]"
        )
    print("[OK] URDF verified.\n")
    return out


if __name__ == "__main__":
    generate_urdf(BatParams())

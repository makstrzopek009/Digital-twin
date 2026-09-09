import os # komunikacja z systemem opracyjnym
import numpy as np
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

from scene import TABLE_SURFACE_Z, HOME_POSE

LOOK_DOWN = euler_angles_to_quats(np.array([0.0, np.pi, np.pi]))
APPROACH_HEIGHT = 0.15

def build_solver():
    ext = get_extension_path_from_name("isaacsim.robot_motion.motion_generation") # sciezka do Luli
    configs = os.path.join(ext, "motion_policy_configs")

    solver = LulaKinematicsSolver(
        robot_description_path = os.path.join(configs, "franka/rmpflow/robot_descriptor.yaml"),     # ktory przegub ma ruszyc i ich limity
        urdf_path = os.path.join(configs, "franka/lula_franka_gen.urdf"),                           # geometrai robota: dl czlonow, pozycje osi
    )

    solver.set_robot_base_pose(
        robot_position = np.array([0.0, 0.0, TABLE_SURFACE_Z]),
        robot_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    )
    return solver

def get_pose(solver, joints, frame ="panda_hand"): # kinematyka prosta

    pos, rot = solver.compute_forward_kinematics(frame,joints)

    return pos, rot

def solve_ik(solver, target_pos, joints, frame = "panda_hand"):
    joints_ik, success = solver.compute_inverse_kinematics(     # Zwraca 7 katow i czy sie udalo
        frame_name = frame,         # ramka ktora ma trafic w cel (panda_hand)
        target_position = target_pos,    # pozcyja docelowa (3 liczby)
        target_orientation = LOOK_DOWN,
        warm_start = joints,        # siedem aktualnych katow
    )

    full = np.array(HOME_POSE)
    full[:7] = joints_ik

    return full, success

def above_target(target):
    return target + np.array([0.0, 0.0, APPROACH_HEIGHT])
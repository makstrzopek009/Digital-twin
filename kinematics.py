import os # komunikacja z systemem opracyjnym
import numpy as np
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

from scene import TABLE_SURFACE_Z, HOME_POSE, ITEM_SIZE, BOX_HEIGHT

LOOK_DOWN = euler_angles_to_quats(np.array([0.0, np.pi, np.pi]))
TCP_OFFSET = 0.1    # odleglosc miedzy panda_hand a srodkiem palcow

APPROACH_HEIGHT = 0.15

GRASP_DEPTH = ITEM_SIZE / 2

# plytszy chwyt
SHALLOW_DEPTH = 0.015
TOP_DEPTH = 0.008     # najplytszy chwyt: palce tylko 2 cm na klocku, dla klockow wystajacych troche ponad sasiadow

# bezpieczna wysokosc TCP przy przejazdach (nad scianka z zapasem na klocek)
SAFE_Z = TABLE_SURFACE_Z + BOX_HEIGHT + 0.10
# dlugosc jednego kroku na prostej [m]
STEP_LEN = 0.1
CARRY_STEP = 0.1

def look_down(yaw):
    # patrzenie w dol + obrot dloni wokol pionu o yaw [rad]
    return euler_angles_to_quats(np.array([0.0, np.pi, np.pi + yaw]))

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

def solve_ik(solver, target_pos, joints, yaw = 0.0, frame = "panda_hand"):

    hand_pos = target_pos + np.array([0.0, 0.0, TCP_OFFSET])

    joints_ik, success = solver.compute_inverse_kinematics(     # Zwraca 7 katow i czy sie udalo
        frame_name = frame,         # ramka ktora ma trafic w cel (panda_hand)
        target_position = hand_pos,    # pozcyja docelowa (3 liczby)
        target_orientation = look_down(yaw),
        warm_start = joints,        # siedem aktualnych katow
    )

    return joints_ik, success

def above_target(target):
    return target + np.array([0.0, 0.0, APPROACH_HEIGHT])

def grasp_target(target, depth = GRASP_DEPTH):
    return target - np.array([0.0, 0.0, depth])

def at_safe(point):
    # ten sam punkt w poziomie, na bezpiecznej wysokosci
    return np.array([point[0], point[1], SAFE_Z])


def line_path(solver, start, end, joints, yaw = 0.0, step_len = STEP_LEN):
    # prosta od start do end pocieta na kroki co STEP_LEN
    # dla kazdego punktu IK, warm start z poprzedniego punktu
    steps = max(1, int(np.ceil(np.linalg.norm(end - start) / step_len))) # liczba krokow na podstawie dl kroku
    path = []
    for i in range(1, steps + 1):
        point = start + (end - start) * i / steps
        joints, ok = solve_ik(solver, point, joints, yaw)
        if not ok:
            return None
        path.append(joints)
    return path
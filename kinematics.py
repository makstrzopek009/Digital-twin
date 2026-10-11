import os # komunikacja z systemem opracyjnym
import numpy as np
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

from scene import TABLE_SURFACE_Z, HOME_POSE, ITEM_SIZE, BOX_HEIGHT
from scipy.spatial.transform import Rotation   # zamiana macierzy obrotu na kwaternion (dziala tez poza Isaakiem)

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
STEP_LEN = 0.02
CARRY_STEP = 0.1

def look_down(yaw):
    # patrzenie w dol + obrot dloni wokol pionu o yaw [rad]
    return euler_angles_to_quats(np.array([0.0, np.pi, np.pi + yaw]))

def hand_quat(n, fingers, alpha = 0.0):
    z = -n                                                   # os podejscia: przeciwnie do normalnej scianki
    y = fingers - (fingers @ z) * z                          # palce prostopadle do osi podejscia
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)                                       # trzecia os dloni
    z = np.cos(alpha) * z + np.sin(alpha) * x                # obrot osi podejscia wokol osi palcow y o kat alfa
    x = np.cross(y, z)                                       # x prostopadle do nowego z
    R = np.column_stack([x, y, z])
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def vertical_alpha(n, fingers):
    z = -n                                                   # os podejscia przy alfa = 0
    y = fingers - (fingers @ z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.arctan2(-x[2], -z[2])                          # alfa, przy ktorym os podejscia jest najblizej pionu (palce poziomo -> dokladnie pion)

def hand_axis(quat):
    # os z dloni (od dloni w strone palcow) w ukladzie swiata, odczytana z kwaternionu (w, x, y, z)
    w, x, y, z = quat
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])   # trzecia kolumna macierzy obrotu

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

def solve_ik(solver, target_pos, joints, quat = LOOK_DOWN, frame = "panda_hand"):

    hand_pos = target_pos - TCP_OFFSET * hand_axis(quat)    # panda_hand lezy TCP_OFFSET za TCP, przeciwnie do osi z dloni

    joints_ik, success = solver.compute_inverse_kinematics(     # Zwraca 7 katow i czy sie udalo
        frame_name = frame,         # ramka ktora ma trafic w cel (panda_hand)
        target_position = hand_pos,    # pozcyja docelowa (3 liczby)
        target_orientation = quat,      # gotowa orientacja dloni (kwaternion)
        warm_start = joints,        # siedem aktualnych katow
    )

    return joints_ik, success

def above_target(target, quat = LOOK_DOWN):
    return target - APPROACH_HEIGHT * hand_axis(quat)       # cofamy sie od celu wzdluz osi dloni

def grasp_target(target, depth = GRASP_DEPTH, quat = LOOK_DOWN):
    return target + depth * hand_axis(quat)                 # wjezdzamy w glab klocka wzdluz osi dloni

def line_path(solver, start, end, joints, quat = LOOK_DOWN, step_len = STEP_LEN):
    # prosta od start do end pocieta na kroki co STEP_LEN
    # dla kazdego punktu IK, warm start z poprzedniego punktu
    steps = max(1, int(np.ceil(np.linalg.norm(end - start) / step_len))) # liczba krokow na podstawie dl kroku
    path = []
    for i in range(1, steps + 1):
        point = start + (end - start) * i / steps
        joints, ok = solve_ik(solver, point, joints, quat)    # ta sama orientacja w kazdym punkcie prostej
        if not ok:
            return None
        path.append(joints)
    return path

def at_safe(point):
    # ten sam punkt w poziomie, na bezpiecznej wysokosci
    return np.array([point[0], point[1], SAFE_Z])

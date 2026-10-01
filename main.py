
# cd C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64
# python.bat C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64\standalone_examples\tutorials\getting_started\main.py

from isaacsim import SimulationApp                  # Przywoluje program z biblioteki

simulation_app = SimulationApp({"headless": False})  # Odpalenie okna z widokiem 3D

import numpy as np
from isaacsim.core.experimental.prims import XformPrim
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd, UsdGeom, UsdPhysics
import isaacsim.core.experimental.utils.stage as stage_utils

import scene
import perception
import kinematics


DETECT = 30   # lb klatek
DROP_EVERY = 15

SETTLE_FRAMES = 30

REMOVE_WALLS = False
walls_removed = False

TARGET_ALL = True # False - najwyzszy klocek, TRUE - wszystkie

# Budowa sceny
franka_robot, items, D435i_sensor = scene.build_scene()


# Solver kinematyki
solver = kinematics.build_solver()
print("RAmki", solver.get_all_frame_names())    # Punkty na robocie, ktory osbluguje solver
pos, rot = kinematics.get_pose(solver, scene.HOME_POSE[:7]) # Bierzemy pozycje przegubow od 0 do 6 (Bez 2 palcow)
print("FK panda_hand:", np.round(pos, 4))

grip_pos, grip_rot = kinematics.get_pose(solver, scene.HOME_POSE[:7], "right_gripper")   # FK ramki right_gripper
print("FK right_gripper:", np.round(grip_pos, 4))                                         # pozycja right_gripper
print("Roznica:", np.round(grip_pos - pos, 4))                                            # right_gripper wzgledem panda_hand
print("Obrot panda_hand:\n", np.round(rot, 3))                                            # macierz obrotu panda_hand
print("Obrot right_gripper:\n", np.round(grip_rot, 3))                                    # macierz obrotu right_gripper

# # pomiar wymiarow palca i dloni
# stage = stage_utils.get_current_stage()
# cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
# for name in ["panda_leftfinger", "panda_hand"]:
#     prim = stage.GetPrimAtPath("/World/Franka/" + name)
#     box = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
#     print(name, "min:", np.round(np.array(box.GetMin()), 4), "max:", np.round(np.array(box.GetMax()), 4))
#     for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
#         if child.IsA(UsdGeom.Gprim):
#             part = cache.ComputeRelativeBound(child, prim).ComputeAlignedRange()
#             print("   ", child.GetName(), "min:", np.round(np.array(part.GetMin()), 4), "max:", np.round(np.array(part.GetMax()), 4))
# lf_pos, _ = kinematics.get_pose(solver, scene.HOME_POSE[:7], "panda_leftfinger")
# print("leftfinger wzgledem panda_hand:", np.round(lf_pos - pos, 4))

rng = np.random.default_rng(0)  # generator liczb losowych // (0) - powtarzalny uklad // (1) - inny powtarzalny uklad // () - randomowo

intrinsics = None
u_tab = None
v_tab = None

# Parametry klatkowe

frame = 0
dropped = 0 # Licznik -  ile zrzuconych
drop_end_frame = 0 # klatka ostatniego zrzutu

# Parametry maszyny stanow

state = "wait"      # co robot robi
state_frame = 0     # ile klatek trwa obecny stan
MOVE_FRAMES = 20
GRIP_FRAMES = 20

# dojechal, gdy najwiekszy blad przegubu mniejszy niz to [rad]
ARRIVE_TOL = 0.01
# co ile klatek kolejny punkt sciezki w prostym przejezdzie
STEP_FRAMES = 3
# ile klatek czekac na swieze zdjecie z nieruchomej kamery
CAM_DELAY = 10

def send_arm(joints):
    # cele tylko dla ramienia, palce bez zmian
    franka_robot.set_dof_position_targets(positions=joints, dof_indices=scene.ARM)


def send_fingers(q):
    # oba palce na ta sama pozycje, ramie bez zmian
    franka_robot.set_dof_position_targets(positions=[q, q], dof_indices=scene.FINGERS)


def arm_at(goal):
    # czy ramie dojechalo do celu
    now = perception.to_numpy(franka_robot.get_dof_positions(dof_indices=scene.ARM)).reshape(-1)
    return np.max(np.abs(now - goal)) < ARRIVE_TOL


def follow(path, step, frame_in_state):
    # co STEP_FRAMES klatek wysyla kolejny punkt sciezki, zwraca numer nastepnego
    if step < len(path) and frame_in_state % STEP_FRAMES == 0:
        send_arm(path[step])
        step += 1
    return step


def path_done(path, step, frame_in_state):
    # wszystkie punkty wyslane i ramie na koncu (albo minal limit czasu)
    if step < len(path):
        return False
    return arm_at(path[-1]) or frame_in_state > len(path) * STEP_FRAMES + MOVE_FRAMES


q_test = kinematics.look_down(np.radians(30))                              # orientacja dloni dla yaw = 30 stopni
R_test = perception.quat_to_matrix(q_test)                                 # ta sama orientacja jako macierz 3x3
fingers = R_test[:, 1]                                                     # os y dloni = kierunek rozsuwania palcow
print("palce przy yaw 30:", round(np.degrees(np.arctan2(fingers[1], fingers[0])) % 180, 1))   # kat palcow w poziomie

# Petla symulacji
while simulation_app.is_running():
    SimulationManager.step()

    state_frame +=1

    '''
    if frame == 120:
        hand = XformPrim("/World/Franka/panda_hand")
        real = perception.to_numpy(hand.get_world_poses()[0])[0]

        print("Scena:", np.round(real, 4))
        print("Lula :", np.round(pos, 4))
        print("Roznica:", np.round(real - pos, 4))
    '''

    depth = perception.read_depth(D435i_sensor) # pobranie glebi i usuniecie dodatkowego wymiaru

    if depth is not None:
  
        h, w = depth.shape[0], depth.shape[1]   # h = wiersze (v), w = kolumny (u)

        if intrinsics is None:
            intrinsics = perception.get_intrinsics(scene.CAMERA_PATH, w, h)
            u_tab, v_tab = perception.make_pixel_grid(w, h)

        if dropped < scene.ITEM_COUNT:
            if frame % DROP_EVERY == 0:

                drop_x = rng.uniform(scene.ITEM_DROP_X_MIN, scene.ITEM_DROP_X_MAX)
                drop_y = rng.uniform(scene.ITEM_DROP_Y_MIN, scene.ITEM_DROP_Y_MAX)
                drop_yaw = rng.uniform(-np.pi / 4, np.pi / 4)
                XformPrim("/World/item{}".format(dropped)).set_world_poses(
                    positions=[[drop_x, drop_y, scene.ITEM_DROP_Z]],
                    orientations=[[np.cos(drop_yaw / 2), 0.0, 0.0, np.sin(drop_yaw / 2)]])   # obrot wokol osi z jako kwaternion

                print("zrzut", dropped, "w", round(drop_x, 3), round(drop_y, 3),"kat:", round(np.degrees(drop_yaw),1))

                dropped += 1

                if dropped == scene.ITEM_COUNT:
                    drop_end_frame = frame

        else:
            if state == "wait": # odczekanie aby klocki sie ustyuowaly na miejscu
                if frame > drop_end_frame + SETTLE_FRAMES:
                    if REMOVE_WALLS and not walls_removed:                       # klocki ulozone
                        for wall in scene.BOX_WALL_PATHS:                        # kazda z czterech scianek
                            XformPrim(wall).set_world_poses(positions=[[5.0, 5.0, 1.0]])   # przesuniecie scianki 5 m od robota, poza stol i poza widok kamery                             # ukrycie: nie widac jej w oknie ani w kamerze
                        walls_removed = True                                     # zeby nie usuwac drugi raz
                        drop_end_frame = frame                                   # odczekaj jeszcze raz, az klocki oparte o scianki sie zsuna
                    else:
                        state = "look"
                        state_frame = -1

            # kilka klatek na swieze zdjecie z nieruchomej kamery
            elif state == "look" and state_frame >= CAM_DELAY and arm_at(np.array(scene.HOME_POSE[:7])):   # zdjecie dopiero gdy ramie stoi w domu
                points = perception.depth_to_pointcloud(depth, u_tab, v_tab, intrinsics, scene.CAMERA_PATH)
                centers, yaws = perception.find_object(points)

                for path_item in scene.item_paths:
                    pos_item, quat_item = XformPrim(path_item).get_world_poses()        # prawdziwa pozycja i obrot z symulacji
                    pos_item = perception.to_numpy(pos_item)[0]                         # odpakowanie z listy
                    quat_item = perception.to_numpy(quat_item)[0] 
                    if abs(pos_item[0] - scene.BOX_CENTER_X) > scene.BOX_INSIDE / 2:    # poza pudelkiem w osi x
                        continue
                    if abs(pos_item[1] - scene.BOX_CENTER_Y) > scene.BOX_INSIDE / 2:    # poza pudelkiem w osi y
                        continue
                    R = perception.quat_to_matrix(quat_item)                            # obrot jako macierz 3x3 (kolumny = osie klocka)
                    edge = R[:, np.argmin(np.abs(R[2, :]))]                             # ta os klocka, ktora lezy najbardziej poziomo
                    true_yaw = np.degrees(np.arctan2(edge[1], edge[0])) % 90            # jej kat w poziomie, 0..90 jak w block_yaw
                    print("prawda", np.round(pos_item, 3), "kat:", round(true_yaw, 1))  # do porownania z linijkami "klocek"


                if len(centers) == 0:
                    print("Pudelko puste")
                    state = "done"
                    state_frame = -1
                else:
                    choice = perception.choose_grasp(points, depth, centers, yaws)

                    if choice is None:
                        print("Brak klockow do chwycenia")
                        state = "done"
                        state_frame = -1
                    else:
                        target, yaw, grip_depth = choice
                        print("Cel:", np.round(target, 3))
                        state = "move"
                        state_frame = -1

            # nad klocek, jeden ruch w przestrzeni przegubow
            elif state == "move":
                if state_frame == 0:
                    above = kinematics.above_target(target)
                    goal, ok = kinematics.solve_ik(solver, above, scene.HOME_POSE[:7], yaw)
                    print("IK:", np.round(goal, 3), "ok", ok)

                    if ok:
                        send_arm(goal)
                        send_fingers(scene.FINGER_PREOPEN)
                    else:
                        print("IK nie znalazlo rozwiazania")
                        state = "done"
                        state_frame = -1
                elif arm_at(goal) or state_frame > MOVE_FRAMES:
                    print("Dojechal", state_frame)
                    state = "down"
                    state_frame = -1

            # zjazd pionowo po prostej do punktu chwytu
            elif state == "down":
                if state_frame == 0:
                    above = kinematics.above_target(target)
                    grasp = kinematics.grasp_target(target, grip_depth)
                    path = kinematics.line_path(solver, above, grasp, goal, yaw)
                    step = 0

                if path is None:
                    print("IK nie znalazlo rozwiazania")
                    state = "done"
                    state_frame = -1
                else:
                    step = follow(path, step, state_frame)
                    if path_done(path, step, state_frame):
                        print("Zjechal", state_frame)
                        goal = path[-1]
                        state = "close"
                        state_frame = -1

            elif state == "close":
                if state_frame == 0:
                    send_fingers(scene.FINGER_CLOSED)
                elif state_frame > GRIP_FRAMES:
                    print("Zacisnal")
                    state = "up"
                    state_frame = -1

            # podjazd pionowo po prostej na bezpieczna wysokosc
            elif state == "up":
                if state_frame == 0:
                    grasp = kinematics.grasp_target(target, grip_depth)
                    lift = kinematics.at_safe(target)
                    path = kinematics.line_path(solver, grasp, lift, goal, yaw)
                    step = 0

                if path is None:
                    print("IK nie znalazlo rozwiazania")
                    state = "open"
                    state_frame = -1
                else:
                    step = follow(path, step, state_frame)
                    if path_done(path, step, state_frame):
                        print("Podjechal", state_frame)
                        goal = path[-1]
                        state = "carry"
                        state_frame = -1

            # przejazd poziomo po prostej na bezpiecznej wysokosci, ten sam yaw
            elif state == "carry":
                if state_frame == 0:
                    start = kinematics.at_safe(target)
                    end = kinematics.at_safe(scene.PLACE_POS)
                    path = kinematics.line_path(solver, start, end, goal, yaw, kinematics.CARRY_STEP)   # przenoszenie dluzszymi krokami
                    step = 0

                if path is None:
                    print("IK nie znalazlo rozwiazania")
                    state = "open"
                    state_frame = -1
                else:
                    step = follow(path, step, state_frame)
                    if path_done(path, step, state_frame):
                        print("Przeniosl", state_frame)
                        goal = path[-1]
                        state = "open"
                        state_frame = -1

            elif state == "open":
                if state_frame == 0:
                    send_fingers(scene.FINGER_OPEN)
                elif state_frame > GRIP_FRAMES:
                    print("Puscil")
                    state = "back"
                    state_frame = -1

            elif state == "back":
                if state_frame == 0:
                    goal = np.array(scene.HOME_POSE[:7])
                    send_arm(goal)
                elif arm_at(goal) or state_frame > MOVE_FRAMES:
                    print("Wrocil", state_frame)
                    state = "look"
                    state_frame = -1
    
 
    frame += 1
    app_utils.update_app()

simulation_app.close()

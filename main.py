# cd C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64
# python.bat C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64\standalone_examples\tutorials\getting_started\main.py

from isaacsim import SimulationApp                  # Przywoluje program z biblioteki

simulation_app = SimulationApp({"headless": False})  # Odpalenie okna z widokiem 3D

import numpy as np
from isaacsim.core.experimental.prims import XformPrim
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation  # kwaternion z katow przy zrzucie
import isaacsim.core.experimental.utils.stage as stage_utils

import scene
import perception
import kinematics


DETECT = 30   # lb klatek
DROP_EVERY = 15

SETTLE_FRAMES = 30

REMOVE_WALLS = True
walls_removed = False



# TEST CZY PODNIOSL
HOLD_WIDTH = scene.ITEM_SIZE * 0.8   # szerokosc chwytu ponad 4 cm = klocek w palcach (klocek ma 5 cm, pusty chwyt ~0)
MAX_MISSES = 3                       # ile pustych chwytow z rzedu, zanim robot sie podda
picked = 0                           # licznik udanych chwytow
missed = 0                           # licznik pustych chwytow (lacznie)
misses_in_row = 0                    # puste chwyty z rzedu
holding = False                      # czy po zacisnieciu klocek jest w palcach

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

# rng = np.random.default_rng(0)  # generator liczb losowych // (0) - powtarzalny uklad // (1) - inny powtarzalny uklad // () - randomowo
SEED = 0                                    # numer ukladu klockow (ten sam seed = ten sam zrzut)
DROP_TILT = 0                 # maksymalne przechylenie klocka przy zrzucie (0 = jak dotad)
rng = np.random.default_rng(SEED)
print("SEED", SEED, "klockow", scene.ITEM_COUNT, "alfa", perception.USE_ALPHA)   # naglowek logu: z jakimi ustawieniami przebieg

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

def grip_width():
    # szerokosc chwytu: suma polozen obu palcow [m]
    q = perception.to_numpy(franka_robot.get_dof_positions(dof_indices=scene.FINGERS)).reshape(-1)   # polozenia przegubow 7 i 8
    return q[0] + q[1]                                                                                  # kazdy palec od srodka dloni, razem szerokosc


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

n_test = np.array([0.0, 0.0, 1.0])                                            # plaska scianka: normalna pionowo w gore
f_test = np.array([np.sin(np.radians(30)), -np.cos(np.radians(30)), 0.0])     # palce tak jak w look_down przy yaw 30
R_new = perception.quat_to_matrix(kinematics.hand_quat(n_test, f_test))       # orientacja z nowej funkcji jako macierz
print("hand_quat vs look_down:", np.abs(R_new - R_test).max())               # najwieksza roznica, powinna byc prawie 0

n_tilt = np.array([0.0, -np.sin(np.radians(25)), np.cos(np.radians(25))])     # scianka przechylona o 25 stopni
R_tilt = perception.quat_to_matrix(kinematics.hand_quat(n_tilt, np.array([1.0, 0.0, 0.0])))   # palce wzdluz x swiata
print("z dloni:", np.round(R_tilt[:, 2], 3), "-n:", np.round(-n_tilt, 3))    # os podejscia powinna byc rowna -n

# Petla symulacji
while simulation_app.is_running():
    
    if frame > 10 and not franka_robot.is_physics_tensor_entity_valid():   # fizyka zatrzymana (np. zamkniete okno) - konczymy petle
        break 
                                                                 # wyjscie z petli, dalej simulation_app.close()
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
            if scene.TEST_RAMPS:                                                      # stanowisko testowe: wszystkie klocki od razu na rampy
                for i, (pos, quat) in enumerate(scene.ramp_item_poses):
                    XformPrim("/World/item{}".format(i)).set_world_poses(positions=[list(pos)], orientations=[quat])
                    print("rampa", i, "kat:", scene.ramp_info[i][0], "kierunek:", scene.ramp_info[i][1], "w", np.round(pos, 3))
                dropped = scene.ITEM_COUNT                                            # zrzut zakonczony
                drop_end_frame = frame                                                # od teraz liczymy czas na ulozenie
            elif frame % DROP_EVERY == 0:

                drop_x = rng.uniform(scene.ITEM_DROP_X_MIN, scene.ITEM_DROP_X_MAX)
                drop_y = rng.uniform(scene.ITEM_DROP_Y_MIN, scene.ITEM_DROP_Y_MAX)
                drop_yaw = rng.uniform(-np.pi / 4, np.pi / 4)
                drop_roll = rng.uniform(-DROP_TILT, DROP_TILT) if DROP_TILT > 0 else 0.0     # przechylenie wokol x
                drop_pitch = rng.uniform(-DROP_TILT, DROP_TILT) if DROP_TILT > 0 else 0.0    # przechylenie wokol y
                q = Rotation.from_euler("xyz", [drop_roll, drop_pitch, drop_yaw]).as_quat()  # kwaternion (x, y, z, w)
                XformPrim("/World/item{}".format(dropped)).set_world_poses(
                    positions=[[drop_x, drop_y, scene.ITEM_DROP_Z]],
                    orientations=[[q[3], q[0], q[1], q[2]]])                                 # Isaac chce (w, x, y, z)

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
                centers, normals, edges = perception.find_object(points)

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
                    R = perception.quat_to_matrix(quat_item)                            # obrot jako macierz 3x3 (kolumny = osie klocka)
                    edge = R[:, np.argmin(np.abs(R[2, :]))]                             # ta os klocka, ktora lezy najbardziej poziomo
                    true_yaw = np.degrees(np.arctan2(edge[1], edge[0])) % 90            # jej kat w poziomie, 0..90 jak w block_yaw
                    up_axis = R[:, np.argmax(np.abs(R[2, :]))]                          # ta os klocka, ktora stoi najbardziej pionowo = normalna gornej scianki
                    true_tilt = np.degrees(np.arccos(np.clip(abs(up_axis[2]), 0.0, 1.0)))   # jej kat od pionu
                    print("prawda", np.round(pos_item, 3), "kat:", round(true_yaw, 1),   # do porownania z linijkami "klocek"
                          "przechyl:", round(true_tilt, 1))                              # prawdziwe przechylenie w stopniach


                if len(centers) == 0:
                    print("Pudelko puste")
                    state = "done"
                    state_frame = -1
                else:
                    choice = perception.choose_grasp(points, depth, centers, normals, edges, solver)

                    if choice is None:
                        print("Brak klockow do chwycenia")
                        state = "done"
                        state_frame = -1
                    else:
                        target, quat, grip_depth = choice                     
                        print("Cel:", np.round(target, 3))
                        state = "move"
                        state_frame = -1

            # nad klocek, jeden ruch w przestrzeni przegubow
            elif state == "move":
                if state_frame == 0:
                    above = kinematics.above_target(target, quat)
                    goal, ok = kinematics.solve_ik(solver, above, scene.HOME_POSE[:7], quat)
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
                    above = kinematics.above_target(target, quat)
                    grasp = kinematics.grasp_target(target, grip_depth, quat)
                    path = kinematics.line_path(solver, above, grasp, goal, quat)
                    step = 0

                if path is None:
                    print("IK nie znalazlo rozwiazania")
                    state = "done"
                    state_frame = -1
                else:
                    step = follow(path, step, state_frame)
                    if path_done(path, step, state_frame):
                        now = perception.to_numpy(franka_robot.get_dof_positions(dof_indices=scene.ARM)).reshape(-1)   # gdzie ramie jest naprawde
                        print("Zjechal", state_frame, "blad [rad]:", round(np.max(np.abs(now - path[-1])), 3))         # ponizej 0.01 = dojechal
                        goal = path[-1]
                        state = "close"
                        state_frame = -1

            elif state == "close":
                if state_frame == 0:
                    send_fingers(scene.FINGER_CLOSED)
                elif state_frame > GRIP_FRAMES:
                    width = grip_width()                                     # jak szeroko zatrzymaly sie palce
                    holding = width > HOLD_WIDTH                             # True = klocek miedzy palcami
                    print("Zacisnal, szerokosc:", round(width, 3), "KLOCEK" if holding else "PUSTO")
                    state = "up"                                             # w gore zawsze, zeby nie ciagnac palcow przez klocki
                    state_frame = -1

            # podjazd pionowo po prostej na bezpieczna wysokosc
            elif state == "up":
                if state_frame == 0:
                    grasp = kinematics.grasp_target(target, grip_depth, quat)
                    lift = kinematics.at_safe(target)
                    path = kinematics.line_path(solver, grasp, lift, goal, quat)
                    step = 0

                if path is None:
                    print("IK nie znalazlo rozwiazania")
                    state = "open"
                    state_frame = -1
                else:
                    step = follow(path, step, state_frame)
                    if path_done(path, step, state_frame):
                        holding = grip_width() > HOLD_WIDTH                  # klocek dalej w palcach? (mogl wypasc przy podnoszeniu)
                        print("Podjechal", state_frame, "KLOCEK" if holding else "PUSTO")
                        if not holding:
                            missed += 1                                      # pusty chwyt albo klocek wypadl przy podnoszeniu
                            misses_in_row += 1
                        goal = path[-1]
                        state = "carry" if holding else "back"               # z klockiem do odlozenia, bez klocka prosto do domu
                        state_frame = -1

            # przejazd poziomo po prostej na bezpiecznej wysokosci, ten sam yaw
            elif state == "carry":
                if state_frame == 0:
                    start = kinematics.at_safe(target)
                    end = kinematics.at_safe(scene.PLACE_POS)
                    path = kinematics.line_path(solver, start, end, goal, quat, kinematics.CARRY_STEP)   # przenoszenie dluzszymi krokami
                    step = 0

                if path is None:
                    print("IK nie znalazlo rozwiazania")
                    state = "open"
                    state_frame = -1
                else:
                    step = follow(path, step, state_frame)
                    if path_done(path, step, state_frame):
                        if grip_width() > HOLD_WIDTH:                        # klocek dojechal w palcach
                            picked += 1                                      # dopiero teraz liczymy klocek jako wyjety
                            misses_in_row = 0
                            print("Przeniosl", state_frame, "DOSTARCZONY")
                        else:
                            missed += 1                                      # klocek wypadl w drodze
                            misses_in_row += 1
                            print("Przeniosl", state_frame, "WYPADL W DRODZE")
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
                    if misses_in_row >= MAX_MISSES:
                        print("Przerywam puste chwyty z rzedu", misses_in_row)
                        state = "done"
                    else:    
                        state = "look"
                    state_frame = -1
            
            elif state == "done":
                if state_frame == 0:                                         # tylko raz, w pierwszej klatce stanu
                    print("KONIEC - wyjete:", picked, "puste chwyty:", missed)
                    if scene.TEST_RAMPS:                                     # podsumowanie stanowiska testowego
                        for i, (angle, direction) in enumerate(scene.ramp_info):
                            pos = perception.to_numpy(XformPrim("/World/item{}".format(i)).get_world_poses()[0])[0]   # gdzie klocek jest teraz
                            on_ramp = np.linalg.norm(pos - scene.ramp_item_poses[i][0]) < 0.03                         # dalej na swojej rampie
                            placed = pos[1] > 0.3                                                                       # strefa odkladania (y ok. 0.5), rampy sa przy |y| < 0.2
                            result = "ODLOZONY" if placed else ("NA RAMPIE" if on_ramp else "STRACONY")
                            print("rampa kat", angle, "kierunek", direction, result)
 
    frame += 1
    app_utils.update_app()

simulation_app.close()

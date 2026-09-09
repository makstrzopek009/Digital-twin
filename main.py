
# cd C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64
# python.bat C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64\standalone_examples\tutorials\getting_started\main.py

from isaacsim import SimulationApp                  # Przywoluje program z biblioteki

simulation_app = SimulationApp({"headless": False})  # Odpalenie okna z widokiem 3D

import numpy as np
from isaacsim.core.experimental.prims import XformPrim
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager


import scene
import perception
import kinematics


DETECT = 30   # lb klatek
DROP_EVERY = 15

SETTLE_FRAMES = 30

TARGET_ALL = True # False - najwyzszy klocek, TRUE - wszystkie

# Budowa sceny
franka_robot, items, D435i_sensor = scene.build_scene()


# Solver kinematyki
solver = kinematics.build_solver()
print("RAmki", solver.get_all_frame_names())    # Punkty na robocie, ktory osbluguje solver
pos, rot = kinematics.get_pose(solver, scene.HOME_POSE[:7]) # Bierzemy pozycje przegubow od 0 do 6 (Bez 2 palcow)
print("FK panda_hand:", np.round(pos, 4))

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
MOVE_FRAMES = 120

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
                XformPrim("/World/item{}".format(dropped)).set_world_poses(
                    positions=[[drop_x, drop_y, scene.ITEM_DROP_Z]])

                print("zrzut", dropped, "w", round(drop_x, 3), round(drop_y, 3))

                dropped += 1

                if dropped == scene.ITEM_COUNT:
                    drop_end_frame = frame

        else:
            if state == "wait":
                if frame > drop_end_frame + SETTLE_FRAMES:
                    state = "look"
                    state_frame = -1

            elif state == "look":
                points = perception.depth_to_pointcloud(depth, u_tab, v_tab, intrinsics, scene.CAMERA_PATH)
                centers = perception.find_object(points)

                if len(centers) > 0:
                    target = max(centers, key=lambda c: c[2])
                    print("Cel:", np.round(target, 3))
                    state = "move"
                    state_frame = -1


            elif state == "move":
                if state_frame == 0:
                    above = kinematics.above_target(target)
                    wynik, ok = kinematics.solve_ik(solver, above, scene.HOME_POSE[:7])
                    print("IK: ", np.round(wynik, 3), "ok", ok)

                    if ok:
                        franka_robot.set_dof_position_targets(positions=wynik)
                    else:
                        print("IK nie znalazlo rozwiazania")
                        state = "look"
                        state_frame = 0
                elif state_frame > MOVE_FRAMES:
                    print("Dojechal")
                    state = "back"
                    state_frame = -1


            elif state == "back":
                if state_frame ==0:
                    franka_robot.set_dof_position_targets(positions = scene.HOME_POSE)
                elif state_frame > MOVE_FRAMES:
                    print("Wrocil")
                    state = "look"
                    state_frame = -1
    
 
    frame += 1
    app_utils.update_app()

simulation_app.close()

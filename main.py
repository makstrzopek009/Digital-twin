
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

DETECT = 30   # lb klatek
DROP_EVERY = 90

# Budowa sceny
franka_robot, items, D435i_sensor = scene.build_scene()

rng = np.random.default_rng(0)  # generator liczb losowych // 0 - powtarzalny uklad

intrinsics = None
u_tab = None
v_tab = None

frame = 0
dropped = 0 # ile zrzuconych

# Pętla symulacji
while simulation_app.is_running():
    SimulationManager.step()

    depth = perception.read_depth(D435i_sensor) # pobranie glebi i usuniecie dodatkowego wymiaru

    if depth is not None:
  
        h, w = depth.shape[0], depth.shape[1]   # h = wiersze (v), w = kolumny (u)

        if intrinsics is None:
            intrinsics = perception.get_intrinsics(scene.CAMERA_PATH, w, h)  # fx, fy, cx, cy
            u_tab, v_tab = perception.make_pixel_grid(w, h)                  # przygotowanie tablic 

        if frame % DETECT == 0 and dropped < scene.ITEM_COUNT:

            drop_x = rng.uniform(scene.ITEM_DROP_X_MIN, scene.ITEM_DROP_X_MAX)
            drop_y = rng.uniform(scene.ITEM_DROP_Y_MIN, scene.ITEM_DROP_Y_MAX)
            
            item_path = "/World/item{}".format(dropped)
            XformPrim(item_path).set_world_poses(positions=[[drop_x, drop_y, scene.ITEM_DROP_Z]])
            print("zrzut", dropped, "w", round(drop_x, 3), round(drop_y, 3))        
            dropped += 1   

            points = perception.depth_to_pointcloud(                         # otrzymujemy pozycje kazdego punktu
                depth, u_tab, v_tab, intrinsics, scene.CAMERA_PATH)
            
            centers = perception.find_object(points)

            if len(centers) > 0:
                target = max(centers, key=lambda c: c[2])
                print("Cel:", np.round(target, 4), " z", len(centers), "obiektow")
            else:
                print("Nie znaleziono klockow")

    frame += 1
    app_utils.update_app()

simulation_app.close()

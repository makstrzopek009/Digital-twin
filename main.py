
# cd C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64
# python.bat C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64\standalone_examples\tutorials\getting_started\main.py

from isaacsim import SimulationApp                  # Przywoluje program z biblioteki

simulation_app = SimulationApp({"headless": False})  # Odpalenie okna z widokiem 3D

import numpy as np
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager

import scene
import perception


DETECT = 30   # lb klatek


# Budowa sceny
franka_robot, items, D435i_sensor = scene.build_scene()

intrinsics = None
u_tab = None
v_tab = None

frame = 0

# Pętla symulacji
while simulation_app.is_running():
    SimulationManager.step()

    depth = perception.read_depth(D435i_sensor) # pobranie glebi i usuniecie dodatkowego wymiaru

    if depth is not None:
  
        h, w = depth.shape[0], depth.shape[1]   # h = wiersze (v), w = kolumny (u)

        if intrinsics is None:
            intrinsics = perception.get_intrinsics(scene.CAMERA_PATH, w, h)  # fx, fy, cx, cy
            u_tab, v_tab = perception.make_pixel_grid(w, h)                  # przygotowanie tablic 

        if frame % DETECT == 0:
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

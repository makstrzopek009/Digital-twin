import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane  # Gotowe obiekty z Isaaca
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
import numpy as np
import omni.kit.viewport.utility as vp_utils

# UsdGeom - odczyt i zapis parametrow optycznych kamery.
# Gf - typy geometryczne USD, potrzebne do ustawienia clippingRange.
from pxr import UsdGeom, Gf

TABLE_SURFACE_Z = 0.75      # gorna powierzchnia blatu
ITEM_SIZE = 0.10            # bok klocka [m]

# Kamera intel realsense D435i
# Wymairy: 90 x 25 x 25
# Zasieg  (Min-Max).3m- 3
# Rozdzielczosc i FPS: 1280x720 30fps
# Katy widzenia: H:87 V:58 horizontal i vertical

CAMERA_PATH = "/World/Franka/panda_hand/robot_camera_D435i"
HFOV_DEG = 87.0             # poziomy kat widzenia D435i
IMG_W, IMG_H = 1280, 720

# Siedem katow przegubow [rad] + dwa polozenia palcow chwytaka [m].
HOME_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]


def configure_camera_optics(camera_path, hfov_deg=HFOV_DEG, near=0.05, far= 10.0):

    prim = stage_utils.get_current_stage().GetPrimAtPath(camera_path)
    usd_camera = UsdGeom.Camera(prim)

    aperture = usd_camera.GetHorizontalApertureAttr().Get()         # przypisanie szerokosci matrycy
    focal = aperture / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))   # obliczenie ogniskowej
    usd_camera.GetFocalLengthAttr().Set(float(focal))               # skonfigurowanie ogniskowej

    # Domyslny clippingRange zaczyna sie na 1.0 m - wszystko blizej zwraca inf.
    usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(float(near), float(far)))    # skonfigurowanie zasięgu


def build_scene():

    # Tworzenie sceny i ustawienie jednostek w metrach
    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)

    # Konfiguracja silnika fizyki (60 Hz)
    # Krok czasowy 0,0166 s silnika fizycznego PhysX
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")

    # Dodanie podłogi i oświetlenia
    GroundPlane("/World/ground_plane")          # sciezka obiektu podlogi
    dome_light = DomeLight("/World/DomeLight")  # sciezka swiatla
    dome_light.set_intensities(1000)

    # Tworzenie stolu
    table_top = Cube(
        paths="/World/TableTop",      # Nadanie sciezki do naszego przedmiotu
        positions=[0.4, 0.0, 0.725],  # SRODEK blatu, 0,4 m przed baza robota
        scales=[0.6, 0.4, 0.025],     # polowy wymiarow -> blat 1.2 x 0.8 x 0.05 m
        colors="red"
    )
    GeomPrim(paths=table_top.paths, apply_collision_apis=True)  # Nalozenie siatki kolizji
    # RigidPrim(paths=table_top.paths)     # Cialo sztywne PhysX,

    # Tworzenie klocka
    item = Cube(
        paths="/World/item",
        positions=[0.5, 0.0, 0.8],    # SRODEK klocka: blat 0.75 + polowa boku 0.05
        scales=[0.05, 0.05, 0.05],    # polowy -> klocek 0.1 x 0.1 x 0.1 m
        colors="blue"
    )

    GeomPrim(paths=item.paths, apply_collision_apis=True)
    RigidPrim(paths=item.paths)     # masa i grawitacja - klocek spada na blat

    # Pobranie pliku robota i dodanie go do sceny
    assets_root = get_assets_root_path()  # Glowny adres z zasobami NVIDIA
    franka_path = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    stage_utils.add_reference_to_stage(usd_path=franka_path, path="/World/Franka")

    # Ustawienie pozycji bazy robota na blacie (używamy XformPrim)
    franka_transform = XformPrim("/World/Franka")
    franka_transform.set_world_poses(positions=[0.0, 0.0, TABLE_SURFACE_Z])  # Ustawiamy robota na blacie

    # Inicjalizacja stawów i silników robota
    franka_robot = Articulation("/World/Franka")


    # Warstwa fizyczna kamery: gdzie stoi i jak jest obrocona
    robot_camera_D435i = RtxCamera(
        path=CAMERA_PATH,
        translations=np.array([0.05, 0.0, 0.04]),     # positions nie dziala lokalnie
        orientations=np.array([0.0, 0.0, 1.0, 0.0]),  # patrzenie tam gdzie chwytak
        tick_rate=30.0
    )

    configure_camera_optics(CAMERA_PATH)

    # Warstwa optyczna kamery: co z niej odczytujemy
    D435i_sensor = CameraSensor(
        robot_camera_D435i,
        resolution=(IMG_H, IMG_W),                  # kolejnosc to (wysokosc, szerokosc)
        annotators=["distance_to_image_plane"]      # odleglosc wzdluz osi optycznej
    )

    # okno podglądu z kamery
    viewport_window = vp_utils.create_viewport_window("Podglad z D435i")
    viewport_window.viewport_api.set_active_camera(CAMERA_PATH)  # nakazujemy programowi przelaczyc sie na okno naszej kamery

    app_utils.play()

    franka_robot.set_dof_position_targets(positions=HOME_POSE)  # Wysyłamy cele do silników PhysX

    return franka_robot, item, D435i_sensor

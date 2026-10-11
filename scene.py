import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane  # Gotowe obiekty z Isaaca
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
import numpy as np
import omni.kit.viewport.utility as vp_utils
from isaacsim.core.experimental.materials import RigidBodyMaterial  # material fizyczny (tarcie)
from pxr import UsdGeom, Gf # UsdGeom - odczyt i zapis parametrow optycznych kamery.
                            # Gf - typy geometryczne USD, potrzebne do ustawienia clippingRange.
from pxr import UsdShade  # przypinanie materialu do prima
from scipy.spatial.transform import Rotation  # obrot rampy z macierzy

# Stol - parametry
TABLE_SURFACE_Z = 0.75      # gorna powierzchnia blatu


# Pudelko - parametry
BOX_CENTER_X = 0.5
BOX_CENTER_Y = 0.0
BOX_INSIDE = 0.4
BOX_WALL = 0.01
BOX_HEIGHT = 0.11 

BOX_WALL_OFFSET = BOX_INSIDE / 2 + BOX_WALL     # dlugosc od srodka sciankli do zewnatrz
BOX_WALL_Z = TABLE_SURFACE_Z + BOX_HEIGHT / 2   # gorna powierzchnia pudelka
BOX_WALL_LENGTH = BOX_INSIDE + 2 * BOX_WALL     # dlugosc pudelka

BOX_WALL_PATHS = ["/World/BoxWallXPlus", "/World/BoxWallXMinus",
                  "/World/BoxWallYPlus", "/World/BoxWallYMinus"]   # sciezki czterech scianek


# Klocek - parametry
ITEM_COUNT = 20
ITEM_SIZE = 0.05           # bok klocka [m]
ITEM_DROP_Z = 0.95
ITEM_DROP_MARGIN = 0.05
ITEM_DROP_X_MIN = BOX_CENTER_X - BOX_INSIDE / 2 + ITEM_DROP_MARGIN
ITEM_DROP_X_MAX = BOX_CENTER_X + BOX_INSIDE / 2 - ITEM_DROP_MARGIN
ITEM_DROP_Y_MIN = BOX_CENTER_Y - BOX_INSIDE / 2 + ITEM_DROP_MARGIN
ITEM_DROP_Y_MAX = BOX_CENTER_Y + BOX_INSIDE / 2 - ITEM_DROP_MARGIN
ITEM_WAIT_X = -1
ITEM_WAIT_Y = -0.8
ITEM_WAIT_STEP = 2 * ITEM_SIZE

wait_z = ITEM_SIZE / 2
item_paths = ["/World/item{}".format(i) for i in range(ITEM_COUNT)]
item_positions = [[ITEM_WAIT_X + i * ITEM_WAIT_STEP, ITEM_WAIT_Y, wait_z]       # Pozycje poczatkowe klockow
                  for i in range(ITEM_COUNT)]


# Stanowisko testowe - rampy o znanym kacie zamiast losowego zrzutu
TEST_RAMPS = False                          # False = zwykly losowy zrzut jak dotad
RAMP_ANGLES = [20, 30, 40]                 # kat rampy [deg]; powyzej 45 szescian sie przewraca
RAMP_DIRS = [0, 90, 180]                   # kierunek spadku [deg]: 0 = od robota (+x), 90 = w bok (+y), 180 = do robota
RAMP_X = [0.40, 0.52, 0.64]                # polozenie w x dla kazdego kata
RAMP_Y = [-0.14, 0.0, 0.14]                # polozenie w y dla kazdego kierunku
RAMP_HALF = 0.05                           # polowa boku rampy [m] -> plyta 10 x 10 cm
RAMP_THICK = 0.01                          # polowa grubosci rampy [m] -> 2 cm
RAMP_TOP_Z = TABLE_SURFACE_Z + 0.035       # wysokosc srodka gornej powierzchni rampy
RAMP_FRICTION = 1.5                        # tarcie rampy, klocek nie zjezdza do ok. 56 st.

# Parametry rampy
def ramp_frame(angle_deg, dir_deg):
    """Obrot rampy: kolumny = w dol stoku, os pochylenia, normalna."""
    th, ph = np.radians(angle_deg), np.radians(dir_deg)
    d = np.array([np.cos(ph), np.sin(ph), 0.0])          # kierunek spadku w poziomie
    up = np.array([0.0, 0.0, 1.0])
    n = np.sin(th) * d + np.cos(th) * up                 # normalna powierzchni rampy (pochylona w strone spadku)
    s = np.cos(th) * d - np.sin(th) * up                 # w dol stoku, w plaszczyznie rampy
    a = np.cross(up, d)                                  # os pochylenia (pozioma)
    return np.column_stack([s, a, n])


def wxyz(R):
    q = Rotation.from_matrix(R).as_quat()                # (x, y, z, w)
    return [q[3], q[0], q[1], q[2]]                      # Isaac chce (w, x, y, z)


ramp_poses = []                                          # (srodek plyty rampy, kwaternion)
ramp_item_poses = []                                     # (srodek klocka, kwaternion) - klocek na srodku rampy
ramp_info = []                                           # (kat, kierunek) w tej samej kolejnosci
for angle, x in zip(RAMP_ANGLES, RAMP_X):
    for direction, y in zip(RAMP_DIRS, RAMP_Y):
        R = ramp_frame(angle, direction)
        top = np.array([x, y, RAMP_TOP_Z])              # srodek gornej powierzchni rampy
        n = R[:, 2]
        ramp_poses.append((top - RAMP_THICK * n, wxyz(R)))                        # plyta pod powierzchnia
        ramp_item_poses.append((top + (ITEM_SIZE / 2 + 0.002) * n, wxyz(R)))     # klocek 2 mm nad rampa, opadnie
        ramp_info.append((angle, direction))

if TEST_RAMPS:
    ITEM_COUNT = len(ramp_item_poses)                    # tyle klockow, ile ramp
    item_paths = ["/World/item{}".format(i) for i in range(ITEM_COUNT)]
    item_positions = [[ITEM_WAIT_X + i * ITEM_WAIT_STEP, ITEM_WAIT_Y, wait_z] for i in range(ITEM_COUNT)]

# Kamera intel realsense D435i
# Wymairy: 90 x 25 x 25
# Zasieg  (Min-Max).3m- 3
# Rozdzielczosc i FPS: 1280x720 30fps
# Katy widzenia: H:87 V:58 horizontal i vertical

CAMERA_PATH = "/World/Franka/panda_hand/robot_camera_D435i"
HFOV_DEG = 87.0             # poziomy kat widzenia D435i
IMG_W, IMG_H = 1280, 720

SHOW_PREVIEW = False

# Siedem katow przegubow [rad] + dwa polozenia palcow chwytaka [m].
HOME_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]

# Numery przegobow 
ARM = [0, 1, 2, 3, 4, 5, 6]
FINGERS = [7, 8]

# Polozenie palcow od srodka dloni [m]

FINGER_OPEN = 0.04
FINGER_PREOPEN = 0.032
FINGER_CLOSED = 0.0


# Chwytak - wymiary [m], zmierzone z modelu (sprawdzic suwmiarka w labie)
# os z dloni: od panda_hand w strone palcow

# od panda_hand do poczatku palca
FINGER_Z = 0.0584
# dlugosc palca (koniec palca 1.23 cm ponizej TCP)
FINGER_LEN = 0.0539
# grubosc palca w kierunku rozsuwania (palec od q do q + FINGER_THICK)
FINGER_THICK = 0.0263
# szerokosc palca w poprzek
FINGER_WIDTH = 0.021

# od panda_hand do spodu dloni (spod dloni 3.4 cm nad TCP)
HAND_Z = 0.066
# polowa dloni w kierunku palcow (wieksza polowa, dlon lekko niesymetryczna)
HAND_HALF_LEN = 0.104
# polowa dloni w poprzek
HAND_HALF_WIDTH = 0.0315

# zapas bezpieczenstwa przy sprawdzaniu kolizji
GRIP_MARGIN = 0.005

# Miejsce odkladania klockow
PLACE_POS = np.array([0.5, 0.5, TABLE_SURFACE_Z + ITEM_SIZE])


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

    # Stanowisko testowe: nieruchome rampy (bez RigidPrim - nie spadaja)
    if TEST_RAMPS:
        ramp_paths = ["/World/Ramp{}".format(i) for i in range(len(ramp_poses))]
        ramps = Cube(
            paths=ramp_paths,
            positions=[list(p) for p, q in ramp_poses],
            scales=[RAMP_HALF, RAMP_HALF, RAMP_THICK],                          # polowy wymiarow -> plyta 10 x 10 x 2 cm
            colors="green"
        )
        for path, (p, q) in zip(ramp_paths, ramp_poses):
            XformPrim(path).set_world_poses(positions=[list(p)], orientations=[q])   # pochylenie rampy
        ramp_geom = GeomPrim(paths=ramps.paths, apply_collision_apis=True)      # siatka kolizji
        ramp_material = RigidBodyMaterial("/World/Materials/ramp")             # material rampy
        ramp_material.set_friction_coefficients(static_frictions=[RAMP_FRICTION], dynamic_frictions=[RAMP_FRICTION])
        ramp_material.set_combine_modes(frictions=["max"])                      # przy styku z klockiem liczy sie tarcie rampy
        ramp_geom.apply_physics_materials(ramp_material)
    
    # Tworzenie scianki prawej
    box_wall_x_plus = Cube(
        paths = "/World/BoxWallXPlus",
        positions = (BOX_CENTER_X + BOX_WALL_OFFSET, BOX_CENTER_Y, BOX_WALL_Z),
        scales = (BOX_WALL / 2, BOX_WALL_LENGTH / 2, BOX_HEIGHT / 2),
        colors = "green"
    )
    GeomPrim(paths=box_wall_x_plus.paths, apply_collision_apis=True)


    # Tworzenie scianki lewej
    box_wall_x_minus = Cube(
        paths = "/World/BoxWallXMinus",
        positions = (BOX_CENTER_X - BOX_WALL_OFFSET, BOX_CENTER_Y, BOX_WALL_Z),
        scales = (BOX_WALL / 2, BOX_WALL_LENGTH / 2, BOX_HEIGHT / 2),
        colors = "green"
    )
    GeomPrim(paths=box_wall_x_minus.paths, apply_collision_apis=True)


    # Tworzenie scianki gornej
    box_wall_y_plus = Cube(
        paths = "/World/BoxWallYPlus",
        positions = (BOX_CENTER_X , BOX_CENTER_Y + BOX_WALL_OFFSET, BOX_WALL_Z),
        scales = (BOX_WALL_LENGTH / 2 - BOX_WALL / 2, BOX_WALL / 2, BOX_HEIGHT / 2),
        colors = "green"
    )
    GeomPrim(paths=box_wall_y_plus.paths, apply_collision_apis=True)


    # Tworzenie scianki dolnej
    box_wall_y_minus = Cube(
        paths = "/World/BoxWallYMinus",
        positions = (BOX_CENTER_X , BOX_CENTER_Y - BOX_WALL_OFFSET, BOX_WALL_Z),
        scales = (BOX_WALL_LENGTH / 2 - BOX_WALL / 2, BOX_WALL / 2, BOX_HEIGHT / 2),
        colors = "green"
    )
    GeomPrim(paths=box_wall_y_minus.paths, apply_collision_apis=True)


    # Tworzenie klocka
    items = Cube(

        paths = item_paths,
        positions = item_positions,
        scales=[ITEM_SIZE /2 ,ITEM_SIZE /2 ,ITEM_SIZE /2 ],    # polowy -> klocek 0.1 x 0.1 x 0.1 m
        colors="blue"
    )

    items_geom = GeomPrim(paths=items.paths, apply_collision_apis=True)                         # siatka kolizji klockow
    grip_material = RigidBodyMaterial("/World/Materials/grip")                                  # nowy material fizyczny
    grip_material.set_friction_coefficients(static_frictions=[1.0], dynamic_frictions=[1.0])    # tarcie wieksze niz domyslne 0.5
    grip_material.set_combine_modes(frictions=["max"])                                          # przy styku liczy sie wieksze z dwoch tarc
    # items_geom.apply_physics_materials(grip_material)                                           # ten sam material dla wszystkich klockow
    RigidPrim(paths=items.paths)     # masa i grawitacja - klocek spada na blat

    # Pobranie pliku robota i dodanie go do sceny
    assets_root = get_assets_root_path()  # Glowny adres z zasobami NVIDIA
    franka_path = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    stage_utils.add_reference_to_stage(usd_path=franka_path, path="/World/Franka")

    # Ustawienie pozycji bazy robota na blacie (używamy XformPrim)
    franka_transform = XformPrim("/World/Franka")
    franka_transform.set_world_poses(positions=[0.0, 0.0, TABLE_SURFACE_Z])  # Ustawiamy robota na blacie
    stage = stage_utils.get_current_stage()                                                # scena USD
    grip = UsdShade.Material(stage.GetPrimAtPath("/World/Materials/grip"))                 # nasz material z tarciem 1.0
    for finger in ["/World/Franka/panda_leftfinger", "/World/Franka/panda_rightfinger"]:   # oba palce
        print("palec", finger, stage.GetPrimAtPath(finger).IsValid())                      # True = sciezka poprawna
        binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(finger))           # pozwala przypiac material do palca
        binding.Bind(grip, UsdShade.Tokens.strongerThanDescendants, "physics")             # material dla wszystkich ksztaltow kolizji palca

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
    # okno podglądu z kamery
    if SHOW_PREVIEW:
        viewport_window = vp_utils.create_viewport_window("Podglad z D435i")
        viewport_window.viewport_api.set_active_camera(CAMERA_PATH)  # nakazujemy programowi przelaczyc sie na okno naszej kamery

    app_utils.play()

    franka_robot.set_dof_position_targets(positions=HOME_POSE)  # Wysyłamy cele do silników PhysX

    return franka_robot, items, D435i_sensor

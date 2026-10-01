import numpy as np
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import XformPrim
from pxr import UsdGeom
from scipy import ndimage # modul do opracji na tablicach wielowymiarowych (mamy funkcje label)

from scene import TABLE_SURFACE_Z, ITEM_SIZE, BOX_CENTER_X, BOX_CENTER_Y, BOX_INSIDE, BOX_HEIGHT

from scene import FINGER_OPEN, FINGER_Z, FINGER_LEN, FINGER_THICK, FINGER_WIDTH, FINGER_PREOPEN
from scene import HAND_Z, HAND_HALF_LEN, HAND_HALF_WIDTH, GRIP_MARGIN
from kinematics import TCP_OFFSET, GRASP_DEPTH, SHALLOW_DEPTH

NOISE_MARGIN = 0.03     # niepewnosc w osi Z [m]
BOX_MARGIN = 0.01     # niepewnosc w osi Z [m]

ROI_X_MIN = BOX_CENTER_X - BOX_INSIDE / 2 + BOX_MARGIN
ROI_X_MAX = BOX_CENTER_X + BOX_INSIDE / 2 - BOX_MARGIN
ROI_Y_MIN = BOX_CENTER_Y - BOX_INSIDE / 2 + BOX_MARGIN
ROI_Y_MAX = BOX_CENTER_Y + BOX_INSIDE / 2 - BOX_MARGIN
ROI_Z_MIN = TABLE_SURFACE_Z + NOISE_MARGIN                # 0.78
ROI_Z_MAX = TABLE_SURFACE_Z + BOX_HEIGHT + NOISE_MARGIN    # 0.75 + 0,11 + 0,03 = 0,89

DEPTH_STEP = 0.001 # skosk wysokosci uznawany za krawedz obiektu


def get_intrinsics(camera_path, width, height):
    """Parametry wewnetrzne kamery: (fx, fy, cx, cy) w pikselach."""

    prim = stage_utils.get_current_stage().GetPrimAtPath(camera_path)
    usd_camera = UsdGeom.Camera(prim)

    focal_length = usd_camera.GetFocalLengthAttr().Get()            # ogniskowa
    horiz_aperture = usd_camera.GetHorizontalApertureAttr().Get()   # szerokosc matrycy
    s = horiz_aperture / width      # szerokosc jednego piksela

    fx = focal_length / s           # ogniskowa wyrazona w pikselach
    fy = fx                         # renderer RTX zaklada kwadratowe piksele

    cx = width / 2.0     # srodek osi optycznej
    cy = height / 2.0
    return fx, fy, cx, cy


def to_numpy(arr):
    """Anotatory moga zwracac tablice warp (na GPU) albo numpy - ujednolicamy."""
    return arr.numpy() if hasattr(arr, "numpy") else np.asarray(arr)


def quat_to_matrix(q):
  
    # Kwaternion (w, x, y, z) -> macierz obrotu 3x3.

    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def get_camera_pose(camera_path):
    """Pozycja kamery w swiecie oraz jej trzy strzalki kierunkowe."""

    pos, quat = XformPrim(camera_path).get_world_poses()  # pozycja, obrot
    pos = to_numpy(pos)[0]      # odpakowanie pos z listy
    quat = to_numpy(quat)[0]    # odpakowanie quat z listy

    R = quat_to_matrix(quat)    # z kwaterionu na tablice wektorow

    right = R[:, 0]     # wszystkie wiersze + pierwsza kolumna
    up = R[:, 1]        # wszystkie wiersze + druga kolumna
    forward = -R[:, 2]  # wszystkei wiersze + trzecia kolumna (-) kamera zwrocona przeciwnie do osi z

    return pos, right, up, forward


def read_depth(sensor):

    depth, info = sensor.get_data("distance_to_image_plane")

    if depth is None:
        return None

    depth = to_numpy(depth)
                                # anotator zwraca H, W, kanal
    if depth.ndim == 3:         # wypakowuwujemy glebokosc bo format jest dostosowany to zdjec RGB
        depth = depth[..., 0]

    return depth


def make_pixel_grid(width, height):

    return np.meshgrid(np.arange(width), np.arange(height))


def depth_to_pointcloud(depth, u_tab, v_tab, intrinsics, camera_path):

    fx, fy, cx, cy = intrinsics

    # KROK 1 - obliczenie X,Y
    X_kam = (u_tab - cx) * depth / fx
    Y_kam = -(v_tab - cy) * depth / fy  # numeracja wierszy w obrazie biegnie odwrotnie niz os kamery

    # KROK 2 - pozycja i kierunek obrotu kamery
    pos, right, up, forward = get_camera_pose(camera_path)

    # [..., None] doklada wymiar, zeby kazda liczba pomnozyla sie przez CALY wektor
    return (pos
            + X_kam[..., None] * right
            + Y_kam[..., None] * up
            + depth[..., None] * forward)


def block_yaw(p):
    """Obrot klocka wokol pionu [rad] i wymiary gornej scianki [m]"""
    xy = p[:, :2]
    xy = xy - xy.mean(axis=0)
    best = None # najlepsza ramka
    for deg in range(90):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        u = xy[:, 0] * c + xy[:, 1] * s          # polozenie punktow wzdluz jednego boku ramki
        v = -xy[:, 0] * s + xy[:, 1] * c         # polozenie punktow wzdluz drugiego boku ramki
        w = u.max() - u.min()                    # szerokosc ramki pod tym katem
        h = v.max() - v.min()                    # dlugosc ramki pod tym katem
        if best is None or w * h < best[3]:      # pierwsza ramka albo mniejsza niz najlepsza dotad?
            best = (a, w, h, w * h)              # zapamietaj: kat, szerokosc, dlugosc, pole

    return best[0], best[1], best[2]             # kat klocka [rad] i wymiary z gory [m]       # kat klocka [rad] i wymiary z gory [m]
            



MIN_PIXELS = 500


def find_object(points):
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    mask = ((z > ROI_Z_MIN) & (z < ROI_Z_MAX)
            & (x > ROI_X_MIN) & (x < ROI_X_MAX)
            & (y > ROI_Y_MIN) & (y < ROI_Y_MAX))
    
    """
test gdzie leza znalezione punkty

masked_points = points[mask]

    print("punktow:", masked_points.shape[0])
    print("x od", round(float(masked_points[:, 0].min()), 3),
          "do", round(float(masked_points[:, 0].max()), 3))
    print("y od", round(float(masked_points[:, 1].min()), 3),
          "do", round(float(masked_points[:, 1].max()), 3))
    print("z od", round(float(masked_points[:, 2].min()), 3),
          "do", round(float(masked_points[:, 2].max()), 3))
    
"""

    step_x = np.abs(np.diff(z, axis=1))   # roznice w poziomie, ksztalt (720, 1279)
    step_y = np.abs(np.diff(z, axis=0))   # roznice w pionie,   ksztalt (719, 1280)

    smooth = np.ones_like(mask, dtype=bool) # tablioca (bez roznic wysokosci, wszystko true)

    smooth[:, :-1] &= (step_x < DEPTH_STEP)   # [:, :-1] to lewo, [:, 1:] to prawo.
    smooth[:, 1:]  &= (step_x < DEPTH_STEP)

    smooth[:-1, :] &= (step_y < DEPTH_STEP)   # [:-1, :] to gora, [1:, :] to gdol
    smooth[1:, :]  &= (step_y < DEPTH_STEP)

    mask = mask & smooth

    labels, count = ndimage.label(mask)

    centers = []
    yaws = []  # akty klockow w tej samej koljnosci co centers

    for i in range(1, count + 1):
        obj_mask = (labels == i)
        if np.count_nonzero(obj_mask) < MIN_PIXELS:
            continue
        p = points[obj_mask]
        centers.append(p.mean(axis=0)) # dopisujemy na koncu
        yaw, w, h = block_yaw(p)                 # kat i wymiary tego klocka z naszej funkcji
        yaws.append(yaw)
        print("klocek", np.round(p.mean(axis=0), 3),       # srodek klocka (x, y, z)
              "kat:", round(np.degrees(yaw), 1),           # kat w stopniach
              "wymiary [cm]:", round(w * 100, 1), round(h * 100, 1))   # wymiary z gory w cm

    return centers, yaws

# Wybor chwytu - parametry

# punkty blizej kamery niz 20 cm to wlasne palce robota
MIN_RANGE = 0.2
# obrot dloni: 0 = palce wzdluz y swiata, pi/2 = palce wzdluz x
GRASP_YAWS = [0.0, np.pi / 2]


def grasp_free(cloud, center, yaw, grip_depth):
    """Czy palce i dlon zmieszcza sie przy chwycie. Zwraca (True/False, powod)."""

    # kierunek rozsuwania palcow i kierunek w poprzek (w poziomie)
    along = np.array([-np.sin(yaw), np.cos(yaw)])               # kierunek palcow, zgodny z obrotem dloni w look_down
    across = np.array([np.cos(yaw), np.sin(yaw)])               # kierunek w poprzek palcow
    # polozenie punktow wzgledem srodka chwytu
    d = cloud[:, :2] - center[:2]
    u = np.abs(d @ along)
    v = np.abs(d @ across)
    z = cloud[:, 2]

    # wysokosc TCP przy chwycie
    tcp_z = center[2] - grip_depth

    # palce: pas od FINGER_OPEN do FINGER_OPEN + FINGER_THICK po obu stronach
    tip_z = tcp_z - (FINGER_Z + FINGER_LEN - TCP_OFFSET)
    in_fingers = ((u > FINGER_PREOPEN - GRIP_MARGIN)
                  & (u < FINGER_PREOPEN + FINGER_THICK + GRIP_MARGIN)
                  & (v < FINGER_WIDTH / 2 + GRIP_MARGIN))
    if np.any(z[in_fingers] > tip_z - GRIP_MARGIN):
        return False, "palec"

    # dlon: prostokat wokol srodka, od spodu dloni w gore
    hand_z = tcp_z + (TCP_OFFSET - HAND_Z)
    in_hand = ((u < HAND_HALF_LEN + GRIP_MARGIN)
               & (v < HAND_HALF_WIDTH + GRIP_MARGIN))
    if np.any(z[in_hand] > hand_z - GRIP_MARGIN):
        return False, "dlon"

    return True, ""


def choose_grasp(points, depth, centers, yaws):
    """Najwyzszy klocek z wolnym chwytem. Zwraca (cel, yaw, glebokosc) albo None."""

    # punkty do sprawdzania: bez wlasnych palcow, bez blatu i podlogi
    valid = np.isfinite(depth) & (depth > MIN_RANGE)
    cloud = points[valid]
    cloud = cloud[cloud[:, 2] > TABLE_SURFACE_Z + GRIP_MARGIN]

    # klocki od najwyzszego
    for center, block in sorted(zip(centers, yaws), key=lambda c: c[0][2], reverse=True):   # klocki od najwyzszego, razem z katem
        reasons = []
        grasp_yaws = sorted([block, block - np.pi / 2], key=abs)        # dwa obroty dloni pasujace do klocka, mniejszy obrot pierwszy




        # najpierw pelna glebokosc, potem plytka; przy kazdej oba obroty
        for grip_depth in [GRASP_DEPTH, SHALLOW_DEPTH]:
            for yaw in grasp_yaws:
                free, reason = grasp_free(cloud, center, yaw, grip_depth)
                if free:
                    print("Chwyt: yaw", round(np.degrees(yaw)), "glebokosc", grip_depth)
                    return center, yaw, grip_depth
                reasons.append(reason)

        print("Pominiety:", np.round(center, 3), reasons)

    return None
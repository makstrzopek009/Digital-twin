import numpy as np
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import XformPrim
from pxr import UsdGeom
from scipy import ndimage # modul do opracji na tablicach wielowymiarowych (mamy funkcje label)

from scene import TABLE_SURFACE_Z, ITEM_SIZE, BOX_CENTER_X, BOX_CENTER_Y, BOX_INSIDE, BOX_HEIGHT


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

    for i in range(1, count + 1):
        obj_mask = (labels == i)
        if np.count_nonzero(obj_mask) < MIN_PIXELS:
            continue
        p = points[obj_mask]
        centers.append(p.mean(axis=0))
    return centers

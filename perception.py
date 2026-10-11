import numpy as np
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import XformPrim
from pxr import UsdGeom
from scipy import ndimage # modul do opracji na tablicach wielowymiarowych (mamy funkcje label)

from scene import TABLE_SURFACE_Z, ITEM_SIZE, BOX_CENTER_X, BOX_CENTER_Y, BOX_INSIDE, BOX_HEIGHT, HOME_POSE

from scene import FINGER_OPEN, FINGER_Z, FINGER_LEN, FINGER_THICK, FINGER_WIDTH, FINGER_PREOPEN
from scene import HAND_Z, HAND_HALF_LEN, HAND_HALF_WIDTH, GRIP_MARGIN
from kinematics import TCP_OFFSET, GRASP_DEPTH, SHALLOW_DEPTH, TOP_DEPTH, APPROACH_HEIGHT, hand_quat, hand_axis, solve_ik, vertical_alpha

USE_ALPHA = True    # False = zawsze podejscie wzdluz -n (do porownania)

NOISE_MARGIN = 0.03     # niepewnosc w osi Z [m]
BOX_MARGIN = -0.05     # niepewnosc w osi Z [m]  bez pudelka - -0.05 //// z pudelkiem - 0.01

ROI_X_MIN = BOX_CENTER_X - BOX_INSIDE / 2 + BOX_MARGIN
ROI_X_MAX = BOX_CENTER_X + BOX_INSIDE / 2 - BOX_MARGIN
ROI_Y_MIN = BOX_CENTER_Y - BOX_INSIDE / 2 + BOX_MARGIN
ROI_Y_MAX = BOX_CENTER_Y + BOX_INSIDE / 2 - BOX_MARGIN
ROI_Z_MIN = TABLE_SURFACE_Z + NOISE_MARGIN                # 0.78
ROI_Z_MAX = TABLE_SURFACE_Z + 0.20 + NOISE_MARGIN    # 0.75 + 0.20 + 0.03 = 0.98 - do 4 warstw klockow (scianki pudelka i tak usuwamy)

DEPTH_STEP = 0.001 # skosk wysokosci uznawany za krawedz obiektu


# Parametery wyznaczania normalnych
NORMAL_ANGLE = 10.0 # [stopnie] roznica normalnych sasiednich pikseli uznawana za krawedz (zalamanie)


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
            
def block_normal(p):
    """Wektor normalny gornej scianki klocka, skierowany w gore"""
    d = p - p.mean(axis=0)                               # punkty wzgledem srodka scianki
    _, _, vt = np.linalg.svd(d, full_matrices=False)     # kierunki rozrzutu punktow, od najwiekszego do najmniejszego
    n = vt[2]                                            # trzeci kierunek = najmniejszy rozrzut = prostopadle do scianki
    if n[2] < 0:                                         # svd moze zwrocic strzalke w dol
        n = -n                                           # odwracamy, zeby patrzyla w gore (w strone kamery)
    return n

def face_frame(p, n):
    """Krawedzie i wymiary scianki liczone w jej wlasnej plaszczyznie (scianka 'polozona plasko')."""
    b1 = np.array([1.0, 0.0, 0.0]) - n[0] * n            # os x swiata rzutowana na plaszczyzne scianki
    b1 = b1 / np.linalg.norm(b1)                         # dlugosc 1
    b2 = np.cross(n, b1)                                 # druga os w plaszczyznie, prostopadla do b1 i do n
    d = p - p.mean(axis=0)                               # punkty wzgledem srodka scianki
    q = np.stack([d @ b1, d @ b2], axis=1)               # wspolrzedne punktow w plaszczyznie scianki (2D)
    a, w, h = block_yaw(q)                               # ta sama ramka co dotad, ale na sciance polozonej plasko
    e1 = np.cos(a) * b1 + np.sin(a) * b2                 # pierwsza krawedz scianki w 3D
    e2 = np.cross(n, e1)                                 # druga krawedz, prostopadla do pierwszej
    return e1, e2, w, h


def pixel_normals(points):
    """Wektor normalny w kazdym pikselu, liczony z sasiada z prawej i z dolu."""
    right = points[:-1, 1:] - points[:-1, :-1]                       # wektor do sasiada z prawej, lezy na powierzchni
    down = points[1:, :-1] - points[:-1, :-1]                        # wektor do sasiada z dolu, tez lezy na powierzchni
    n = np.cross(right, down)                                        # iloczyn wektorowy: prostopadly do obu = normalna
    n = n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-12)       # dlugosc 1 (1e-12 chroni przed dzieleniem przez zero)
    n[n[..., 2] < 0] *= -1                                           # wszystkie strzalki w gore
    normals = np.zeros_like(points)                                  # pelny rozmiar obrazu, ostatni wiersz i kolumna zostaja zerami
    normals[:-1, :-1] = n                                            # wpisujemy policzone normalne
    return normals

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

    nrm = pixel_normals(points)                          # normalna w kazdym pikselu
    cos_x = np.sum(nrm[:, 1:] * nrm[:, :-1], axis=2)     # cos kata miedzy normalna piksela i sasiada z prawej, ksztalt (720, 1279)
    cos_y = np.sum(nrm[1:, :] * nrm[:-1, :], axis=2)     # cos kata z sasiadem z dolu, ksztalt (719, 1280)
    min_cos = np.cos(np.radians(NORMAL_ANGLE))           # granica 10 stopni zapisana jako cosinus

    smooth[:, :-1] &= (cos_x > min_cos)                  # zalamanie miedzy pikselem a sasiadem z prawej
    smooth[:, 1:]  &= (cos_x > min_cos)
    smooth[:-1, :] &= (cos_y > min_cos)                  # zalamanie miedzy pikselem a sasiadem z dolu
    smooth[1:, :]  &= (cos_y > min_cos)

    mask = mask & smooth

    labels, count = ndimage.label(mask)

    centers = []
    normals = []  # normalne scianek w tej samej kolejnosci co centers
    edges = []    # pary krawedzi (e1, e2) w tej samej kolejnosci

    for i in range(1, count + 1):
        obj_mask = (labels == i)
        if np.count_nonzero(obj_mask) < MIN_PIXELS:
            continue
        p = points[obj_mask]
        centers.append(p.mean(axis=0)) # dopisujemy na koncu
        yaw, w, h = block_yaw(p)                 # kat i wymiary tego klocka z naszej funkcji
        n = block_normal(p)                                      # wektor normalny gornej scianki
        tilt = np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0)))   # przechylenie: kat miedzy normalna a pionem
        e1, e2, w3, h3 = face_frame(p, n)                        # krawedzie i prawdziwe wymiary scianki
        normals.append(n)                                        # zapamietaj normalna tej scianki
        edges.append((e1, e2))                                   # i jej dwie krawedzie
        print("klocek", np.round(p.mean(axis=0), 3),       # srodek klocka (x, y, z)
              "kat:", round(np.degrees(yaw), 1),           # kat w stopniach
              "wymiary [cm]:", round(w * 100, 1), round(h * 100, 1),   # wymiary z gory w cm
              "przechyl:", round(tilt, 1),                 # przechylenie scianki w stopniach
              "w plaszczyznie [cm]:", round(w3 * 100, 1), round(h3 * 100, 1))   # wymiary scianki polozonej plasko

    return centers, normals, edges

# Wybor chwytu - parametry

# punkty blizej kamery niz 20 cm to wlasne palce robota
MIN_RANGE = 0.2

# przesuniecie chwytu
GRASP_SHIFT = 0.013

MAX_JOINT_JUMP = 1.5    # [rad] przeguby 1-6 dalej od domu = ramie przekreca sie przez cala scene
MAX_DESCENT_TURN = 1.0    # [rad] tyle moga sie obrocic przeguby 1-6 w czasie zjazdu (normalnie 0.3-0.8)


# obrot dloni: 0 = palce wzdluz y swiata, pi/2 = palce wzdluz x
GRASP_YAWS = [0.0, np.pi / 2]


def grasp_free(cloud, center, quat, grip_depth):
    """Czy palce i dlon zmieszcza sie przy chwycie. Zwraca (True/False, powod)."""

    R = quat_to_matrix(quat)                     # osie dloni w ukladzie swiata (kolumny macierzy)
    across = R[:, 0]                             # x_d: w poprzek palcow
    along = R[:, 1]                              # y_d: kierunek rozsuwania palcow
    axis = R[:, 2]                               # z_d: od dloni w strone palcow, w glab klocka

    # polozenie punktow wzgledem TCP, w osiach dloni
    tcp = center + grip_depth * axis             # TCP przy chwycie (to samo co grasp_target)
    d = cloud - tcp
    u = np.abs(d @ along)                        # odleglosc wzdluz palcow
    v = np.abs(d @ across)                       # odleglosc w poprzek palcow
    b = -(d @ axis)                              # wysokosc nad TCP, liczona wzdluz osi dloni w strone dloni

    # palce: pas od FINGER_PREOPEN do FINGER_PREOPEN + FINGER_THICK po obu stronach
    tip_b = -(FINGER_Z + FINGER_LEN - TCP_OFFSET)          # koniec palca wzgledem TCP (1.23 cm pod TCP)
    in_fingers = ((u > FINGER_PREOPEN - GRIP_MARGIN)
                  & (u < FINGER_PREOPEN + FINGER_THICK + GRIP_MARGIN)
                  & (v < FINGER_WIDTH / 2 + GRIP_MARGIN))
    hit = b[in_fingers] > tip_b - GRIP_MARGIN                     # ktore punkty w obszarze palca sa powyzej jego konca
    if np.any(hit):
        top = cloud[in_fingers][hit][:, 2].max()                   # najwyzszy taki punkt, wysokosc w swiecie [m]
        return False, "palec %.3f" % top                           # 0.75 = blat, ok. 0.80 = gora plaskiego sasiada

    # dlon: prostokat wokol srodka, od spodu dloni w gore
    hand_b = TCP_OFFSET - HAND_Z                            # spod dloni wzgledem TCP (3.4 cm nad TCP)
    in_hand = ((u < HAND_HALF_LEN + GRIP_MARGIN)
               & (v < HAND_HALF_WIDTH + GRIP_MARGIN))
    if np.any(b[in_hand] > hand_b - GRIP_MARGIN):
        return False, "dlon"

    return True, ""


def choose_grasp(points, depth, centers, normals, edges, solver):
    """Najwyzsza scianka z wolnym i osiagalnym chwytem. Zwraca (cel, kwaternion, glebokosc) albo None."""

    # punkty do sprawdzania: bez wlasnych palcow, blat zostaje
    valid = np.isfinite(depth) & (depth > MIN_RANGE)
    cloud = points[valid]
    cloud = cloud[cloud[:, 2] > TABLE_SURFACE_Z - GRIP_MARGIN]

    # scianki od najwyzszej
    for center, n, edge in sorted(zip(centers, normals, edges), key=lambda c: c[0][2], reverse=True):
        reasons = []
        e1, e2 = edge[0], edge[1]                                                  # krawedzie scianki
        tilt = np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0)))                     # przechylenie scianki

        fingers = [f if f[1] <= 0 else -f for f in (e1, e2)]                       # palce w strone -y
        fingers.sort(key=lambda f: f[1])                                           # najpierw bardziej ujemne y

        for grip_depth in [GRASP_DEPTH, SHALLOW_DEPTH, TOP_DEPTH]:
            for f in fingers:
                across = quat_to_matrix(hand_quat(n, f))[:, 0]                     # x_d przy alfa = 0: wzdluz chwytanych scian, w plaszczyznie scianki
                best = vertical_alpha(n, f)                                        # alfa najblizej pionu
                alphas = [best, best / 2, 0.0] if USE_ALPHA and abs(best) > np.radians(1) else [0.0]   # najpierw pion, potem posrednio, na koncu wzdluz -n

                for alpha in alphas:
                    quat = hand_quat(n, f, alpha)                                  # dlon obrocona o alfa wokol osi palcow
                    axis = hand_axis(quat)                                         # kierunek podejscia z_d po obrocie

                    for shift in [0.0, GRASP_SHIFT, -GRASP_SHIFT]:
                        tcp = center + shift * across - grip_depth * n             # TCP: glebokosc od scianki wzdluz -n, jak dotad (5 cm klocek -> srodek)
                        point = tcp - grip_depth * axis                            # cel tak, zeby grasp_target(point) trafil w ten sam TCP
                        free, reason = grasp_free(cloud, point, quat, grip_depth)  # czy palce i dlon sie zmieszcza
                        if free:                                                   # palce sie mieszcza, czy ramie siegnie?
                            above = point - APPROACH_HEIGHT * axis                 # punkt nad chwytem, jak w stanie move
                            q_ik, reach = solve_ik(solver, above, np.array(HOME_POSE[:7]), quat)   # IK z pozycji domowej
                            jump = np.max(np.abs(q_ik[:6] - np.array(HOME_POSE[:6])))             # najwiekszy obrot przegubow 1-6 od domu (7 to tylko obrot dloni)
                            if not reach or jump > MAX_JOINT_JUMP:
                                free, reason = False, "ik"                         # poza zasiegiem albo dziwne ulozenie ramienia, szukamy dalej
                            else:
                                low = point + grip_depth * axis                    # koniec zjazdu (TCP przy chwycie)
                                q_low, reach = solve_ik(solver, low, q_ik, quat)  # IK na dole, start z gornego rozwiazania
                                turn = np.max(np.abs(q_low[:6] - q_ik[:6]))      # ile przeguby musza sie obrocic w czasie zjazdu
                                if not reach or turn > MAX_DESCENT_TURN:
                                    free, reason = False, "ik dol"                 # zjazd wymaga przekrecenia ramienia, szukamy dalej                        # poza zasiegiem, szukamy dalej
                        if free:
                            print("Chwyt: przechyl", round(tilt), "palce", np.round(f, 2),
                                  "alfa", round(np.degrees(alpha)),
                                  "glebokosc", grip_depth, "przesuniecie", shift)
                            return point, quat, grip_depth
                        reasons.append(reason)

        print("Pominiety:", np.round(center, 3), sorted(set(reasons)))

    return None
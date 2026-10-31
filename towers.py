"""Ukladanie wiez w rzedzie."""

import numpy as np
import itertools
from scipy.spatial.transform import Rotation
import scene
import perception
import kinematics
from scene import TABLE_SURFACE_Z, ITEM_SIZE, HOME_POSE
from scene import GRIP_MARGIN, FINGER_OPEN, FINGER_Z, FINGER_LEN, FINGER_THICK, FINGER_WIDTH
from scene import HAND_Z, HAND_HALF_LEN, HAND_HALF_WIDTH
from kinematics import TCP_OFFSET, LOOK_DOWN, SAFE_Z, solve_ik, tcp_pose

VIEW_DIST = 0.27      # kamera tyle nad gora pelnej wiezy przy ogladaniu rzedu [m]
SNAP = 0.03           # gora wiezy szukana w tym promieniu od jej ostatniego srodka [m]
PLACE_GAP = 0.004                  # klocek puszczany 4 mm nad gora wiezy [m]
MAX_PLACE_TILT = np.radians(55)    # najwieksze odchylenie dloni od pionu przy odkladaniu (|beta| do 45 st. + zapas)
TURN_WEIGHT = 0.2                  # waga obrotu dloni miedzy chwytem a odlozeniem w koszcie
ROW_PENALTY = 20.0                 # kara [deg] za palce wzdluz rzedu (dlon nad sasiednimi wiezami)
BUILT_PENALTY = 30.0               # kara [deg] za pochylenie dloni w strone gotowej wiezy

# bryly w tescie slupa (palce i dlon ze scene.py, reszta przyblizona)
PALM_THICK = 0.046                 # grubosc dloni [m]
WRIST_HALF = 0.045                 # polowa szerokosci nadgarstka nad dlonia [m]
WRIST_TOP = 0.20                   # nadgarstek sprawdzany do tej wysokosci nad TCP [m]
BLOCK_MARGIN = 0.002               # zapas wokol odkladanego klocka [m] (musi byc mniej niz PLACE_GAP)
NEAR_XY = 0.35                     # do testu punkty blizej niz tyle od miejsca odlozenia (w poziomie) [m]
SUPPORT_TOL = 0.01                 # punkty nizej niz tyle nad gora wiezy = sama wieza lub blat (blad kamery kilka mm) [m]
TRANSIT_MARGIN = 0.03              # przelot: spod klocka i palcow tyle nad najwyzszym punktem sceny [m]
TRANSIT_X_MIN = 0.15               # najwyzszy punkt sceny liczony od tego x (blizej stoi podstawa robota) [m]
MAX_FAILS = 3                      # tyle razy z rzedu miejsce zajete (nawet dla prostego chwytu) = wieza zamknieta
CHECK_XY = 0.005                   # kontrola po odlozeniu: przesuniecie gory ponad to = odchylka [m]
CHECK_Z = 0.005                    # i blad wysokosci ponad to = odchylka [m]
CLOSE_XY = 0.015                   # przesuniecie gory ponad to = wieza zamknieta [m]
CLOSE_TILT = np.radians(3)         # przechylenie gory ponad to = wieza zamknieta
pending = None                     # ostatnie odlozenie czekajace na kontrole na zdjeciu wiez
FIELD_PAD = 0.15                   # chmura pola wiez: pas tej szerokosci wokol rzedu [m]
view_checked = False  # kadr sprawdzony przy pierwszym zdjeciu wiez

slots = []    # stan wiez, w kolejnosci budowy
field_cloud = np.zeros((0, 3))     # punkty pola wiez z ostatniego zdjecia wiez


def init():
    """Wieze w kolejnosci budowy: najpierw najdalsza od robota, wtedy ramie nie przechodzi nad gotowymi."""
    global slots
    order = sorted(scene.TOWER_SLOTS, key=lambda xy: -np.linalg.norm(xy))     # od najdalszej od podstawy robota
    slots = [{"nr": k + 1,                    # numer wiezy w kolejnosci budowy
              "xy": xy.copy(),                # srodek miejsca na wieze
              "top_xy": xy.copy(),            # srodek gory wiezy (z kamery), na poczatku srodek miejsca
              "top_z": TABLE_SURFACE_Z,       # wysokosc gory wiezy, na poczatku blat
              "yaw": scene.TOWER_YAW,         # obrot gory wiezy wokol pionu
              "tilt": 0.0,                    # przechylenie gory wiezy
              "count": 0,                     # ile klockow stoi na wiezy
              "closed": False,                # wieza zamknieta (krzywa albo przewrocona) - nie dokladamy
              "fails": 0}                     # ile razy z rzedu miejsce bylo zajete
             for k, xy in enumerate(order)]
    check_layout()


def check_layout():
    """Czy wieze stoja na stole i poza obszarem chwytania."""
    half = ITEM_SIZE / 2
    x0 = scene.TABLE_CENTER[0] - scene.TABLE_HALF[0]           # brzegi blatu
    x1 = scene.TABLE_CENTER[0] + scene.TABLE_HALF[0]
    y0 = scene.TABLE_CENTER[1] - scene.TABLE_HALF[1]
    y1 = scene.TABLE_CENTER[1] + scene.TABLE_HALF[1]
    print("Wieze (kolejnosc budowy, x, y):", [(s["nr"], np.round(s["xy"], 3).tolist()) for s in slots],
          "po", scene.TOWER_HEIGHT, "klockow")
    for s in slots:
        x, y = s["xy"]
        if x - half < x0 or x + half > x1 or y - half < y0 or y + half > y1:      # klocek wystaje poza blat
            print("UWAGA: wieza", s["nr"], "poza stolem")
        if (x + half > perception.ROI_X_MIN and x - half < perception.ROI_X_MAX     # kwadrat klocka zachodzi
                and y + half > perception.ROI_Y_MIN and y - half < perception.ROI_Y_MAX):   # na obszar chwytania
            print("UWAGA: wieza", s["nr"], "w obszarze chwytania - robot moze z niej zdejmowac klocki")

def field_box(pad):
    """Prostokat wokol srodkow wiez, poszerzony o pad [m]: (x min, y min), (x max, y max)."""
    xy = np.array([s["xy"] for s in slots])
    return xy.min(axis=0) - pad, xy.max(axis=0) + pad


def view_camera():
    """Gdzie ma stac kamera przy ogladaniu wiez: pionowo nad srodkiem rzedu, VIEW_DIST nad gora pelnej wiezy."""
    lo, hi = field_box(0.0)
    return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                     TABLE_SURFACE_Z + scene.TOWER_HEIGHT * ITEM_SIZE + VIEW_DIST])


def view_target():
    """TCP, przy ktorym kamera stoi w view_camera()."""
    R = perception.quat_to_matrix(LOOK_DOWN)              # osie dloni patrzacej w dol
    hand = view_camera() - R @ scene.CAMERA_OFFSET        # panda_hand: cofamy przesuniecie kamery
    return hand + TCP_OFFSET * R[:, 2]                    # TCP lezy 10 cm przed panda_hand wzdluz osi dloni


def view_joints(solver):
    """Katy przegubow w pozycji ogladania wiez (None = poza zasiegiem)."""
    q, ok = solve_ik(solver, view_target(), np.array(HOME_POSE[:7]), LOOK_DOWN)   # start z pozycji domowej
    return q if ok else None
def field_roi():
    """Obszar segmentacji wiez: rzad + klocek zapasu, od 3 cm nad blatem do gory pelnej wiezy."""
    lo, hi = field_box(ITEM_SIZE)
    return (lo[0], hi[0], lo[1], hi[1], TABLE_SURFACE_Z + perception.NOISE_MARGIN,
            TABLE_SURFACE_Z + scene.TOWER_HEIGHT * ITEM_SIZE + perception.NOISE_MARGIN)


def check_view(intrinsics, width, height):
    """Czy gora pelnej wiezy kazdej wiezy miesci sie w kadrze z pozycji ogladania."""
    z_top = TABLE_SURFACE_Z + scene.TOWER_HEIGHT * ITEM_SIZE
    c = perception.camera_footprint(intrinsics, scene.CAMERA_PATH, width, height, z_top)
    lo, hi = c[:, :2].min(axis=0), c[:, :2].max(axis=0)                  # kamera patrzy pionowo: kadr to prostokat
    out = [s["nr"] for s in slots
           if np.any(s["xy"] - ITEM_SIZE / 2 < lo) or np.any(s["xy"] + ITEM_SIZE / 2 > hi)]
    print("Kadr wiez na wysokosci pelnej wiezy: x", np.round([lo[0], hi[0]], 3).tolist(),
          "y", np.round([lo[1], hi[1]], 3).tolist(), "OK" if not out else "UWAGA: poza kadrem wieze %s" % out)


def update(points, depth, intrinsics):
    """Stan wiez ze zdjecia z pozycji ogladania."""
    global view_checked, field_cloud, pending
    if not view_checked:                                                 # kadr sprawdzamy raz, przy pierwszym zdjeciu
        check_view(intrinsics, depth.shape[1], depth.shape[0])
        view_checked = True
    cloud = perception.collision_cloud(points, depth)                    # punkty bez wlasnych palcow
    lo, hi = field_box(FIELD_PAD)                                        # prostokat: rzad +-15 cm
    keep = (cloud[:, 0] > lo[0]) & (cloud[:, 0] < hi[0]) & (cloud[:, 1] > lo[1]) & (cloud[:, 1] < hi[1])
    field_cloud = cloud[keep][::4]                                       # co czwarty punkt wystarczy do testow kolizji
    faces = perception.segment(points, field_roi())
    for s in slots:
        near = [f for f in faces if np.linalg.norm(f["center"][:2] - s["top_xy"]) < SNAP]
        if near:
            top = max(near, key=lambda f: f["center"][2])                # najwyzsza scianka przy wiezy = jej gora
            s["top_xy"] = top["center"][:2].copy()
            s["top_z"] = top["center"][2]
            s["yaw"] = (np.arctan2(top["e1"][1], top["e1"][0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4   # -45..45 st.
            s["tilt"] = np.arccos(np.clip(top["n"][2], -1.0, 1.0))
            s["count"] = int(round((s["top_z"] - TABLE_SURFACE_Z) / ITEM_SIZE))
        else:                                                            # nic nie stoi: puste miejsce
            s["top_xy"], s["top_z"], s["yaw"], s["tilt"], s["count"] = s["xy"].copy(), TABLE_SURFACE_Z, scene.TOWER_YAW, 0.0, 0
        print("Wieza", s["nr"], "klockow %d/%d" % (s["count"], scene.TOWER_HEIGHT),
              "gora", np.round([*s["top_xy"], s["top_z"]], 3).tolist(),
              "obrot", round(np.degrees(s["yaw"]), 1), "przechyl", round(np.degrees(s["tilt"]), 1))
    if pending is not None:                                              # kontrola ostatniego odlozenia
        s = slots[pending["slot"]]
        dxy = np.linalg.norm(s["top_xy"] - pending["xy"])               # przesuniecie gory wzgledem planu
        dz = s["top_z"] - pending["z"]                                  # blad wysokosci gory
        if s["count"] < pending["count"] - 1:
            result, s["closed"] = "PRZEWROCONA - wieza zamknieta", True
        elif s["count"] < pending["count"]:
            result = "KLOCEK SPADL Z WIEZY"
        elif s["tilt"] > CLOSE_TILT:
            result, s["closed"] = "KRZYWO - wieza zamknieta", True
        elif dxy > CLOSE_XY:
            result, s["closed"] = "ZA DALEKO OD PLANU - wieza zamknieta", True
        elif dxy > CHECK_XY or abs(dz) > CHECK_Z:
            result = "OK z odchylka"
        else:
            result = "OK"
        print("Kontrola wiezy", s["nr"], "warstwa", pending["count"], "przesuniecie [mm]:", round(dxy * 1000, 1),
              "wysokosc [mm]:", round(dz * 1000, 1), "przechyl:", round(np.degrees(s["tilt"]), 1), result)
        pending = None                                                   # sprawdzone

def cube_rotations():
    """24 obroty, po ktorych szescian wyglada tak samo (macierze permutacji ze znakami, wyznacznik +1)."""
    out = []
    for perm in itertools.permutations(range(3)):                   # ktora os idzie na ktora (6 mozliwosci)
        for signs in itertools.product((1.0, -1.0), repeat=3):      # zwroty osi (8 mozliwosci)
            M = np.zeros((3, 3))
            for col, (row, sg) in enumerate(zip(perm, signs)):
                M[row, col] = sg
            if np.linalg.det(M) > 0:                                # tylko obroty, bez odbic lustrzanych
                out.append(M)
    return out


CUBE = cube_rotations()


def rot_angle(M):
    # kat obrotu opisanego macierza M [rad]: slad macierzy obrotu = 1 + 2 cos(kat)
    return np.arccos(np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0))


def block_in_hand(quat, tcp, block):
    """Jak klocek lezy w dloni: obrot i srodek klocka w osiach dloni (stale, jesli sie nie obraca w palcach)."""
    R_d = perception.quat_to_matrix(quat)
    b, R_k = block
    return R_d.T @ R_k, R_d.T @ (b - tcp)


def in_hand_check(solver, joints, R_kd, p_kd, item_pos, item_rot):
    """Przewidziany klocek w dloni wobec prawdy z symulatora (tylko pomiar)."""
    tcp, R = tcp_pose(solver, joints)                                   # gdzie jest dlon naprawde
    R_pred = R @ R_kd                                                   # przewidziany obrot klocka
    b_pred = tcp + R @ p_kd                                             # przewidziany srodek klocka
    i = np.argmin(np.linalg.norm(item_pos - b_pred, axis=1))            # klocek najblizej tego miejsca
    ang = min(rot_angle((R_pred @ S).T @ item_rot[i]) for S in CUBE)    # blad obrotu z dokladnoscia do symetrii
    print("Klocek w dloni: blad obrotu [deg]:", round(np.degrees(ang), 1),
          "polozenia [mm]:", round(np.linalg.norm(item_pos[i] - b_pred) * 1000, 1))

def rot_z(a):
    # obrot o kat a wokol pionu
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def to_quat(R):
    # macierz obrotu -> kwaternion (w, x, y, z) jak w reszcie kodu
    q = Rotation.from_matrix(R).as_quat()                     # scipy daje (x, y, z, w)
    return np.array([q[3], q[0], q[1], q[2]])


def free_slot():
    """Numer (indeks) wiezy, ktora teraz budujemy: pierwsza w kolejnosci, niepelna i niezamknieta."""
    for i, s in enumerate(slots):
        if not s["closed"] and s["count"] < scene.TOWER_HEIGHT:
            return i
    return None


def place_candidates(s, R_d, R_kd, p_kd):
    """Ustawienia dloni przy odkladaniu na wieze s, od najtanszego. Zwraca (lista, srodek klocka nad wieza)."""
    b_new = np.array([s["top_xy"][0], s["top_xy"][1], s["top_z"] + ITEM_SIZE / 2 + PLACE_GAP])
    row = slots[-1]["xy"] - slots[0]["xy"] if len(slots) > 1 else np.array([1.0, 0.0])
    row = row / (np.linalg.norm(row) + 1e-12)                             # kierunek rzedu
    built = [o["xy"] for o in slots if o is not s and o["count"] > 0
             and np.linalg.norm(o["xy"] - s["xy"]) < 1.5 * scene.TOWER_GAP]   # zbudowani sasiedzi

    cands = []
    for S in CUBE:                                                        # 24 ustawienia klocka
        R_k_new = rot_z(s["yaw"]) @ S                                     # klocek poziomo, krawedzie jak gora wiezy
        R_new = R_k_new @ R_kd.T                                          # dlon, ktora trzyma klocek w takim ustawieniu
        tilt = np.arccos(np.clip(-R_new[2, 2], -1.0, 1.0))                # odchylenie osi dloni od pionu w dol
        if tilt > MAX_PLACE_TILT:
            continue
        cost = np.degrees(tilt) + TURN_WEIGHT * np.degrees(rot_angle(R_d.T @ R_new))
        if abs(R_new[:2, 1] @ row) > 0.7:                                 # palce wzdluz rzedu
            cost += ROW_PENALTY
        lean = -R_new[:2, 2]                                              # w ktora strone przechyla sie dlon (od palcow do dloni)
        if np.degrees(tilt) > 5 and any(lean @ (xy - s["xy"]) > 1e-3 for xy in built):
            cost += BUILT_PENALTY                                         # dlon pochylona w strone gotowej wiezy
        tcp_new = b_new - R_new @ p_kd                                    # TCP przy odkladaniu
        cands.append((cost, tilt, R_k_new, R_new, tcp_new))
    cands.sort(key=lambda c: c[0])
    return cands, b_new

def hang(R, R_k, p):
    """Ile klocek i zacisniete palce wystaja pod TCP [m]. R - dlon, R_k - klocek, p - srodek klocka wzgledem TCP."""
    zero = np.zeros(3)                                                    # TCP w poczatku ukladu: liczymy wzgledem niego
    return -lowest([(p, R_k, np.full(3, ITEM_SIZE / 2))] + finger_boxes(R, zero, ITEM_SIZE / 2))


def check_blocked(s, near, b_new):
    """Czy miejsce na wiezy jest zajete nawet dla najprostszego chwytu. MAX_FAILS razy z rzedu = wieza zamknieta."""
    R_ref = rot_z(s["yaw"]) @ np.diag([1.0, -1.0, -1.0])                 # dlon pionowo w dol, palce w poprzek rzedu
    boxes = (finger_boxes(R_ref, b_new, FINGER_OPEN) + body_boxes(R_ref, b_new)
             + [(b_new, rot_z(s["yaw"]), np.full(3, ITEM_SIZE / 2))])     # TCP w srodku klocka, klocek plasko
    if column_hit(near, boxes, [GRIP_MARGIN] * 4 + [BLOCK_MARGIN]) is None:
        return                                                            # miejsce wolne - winny byl tylko ten chwyt
    s["fails"] += 1
    print("Miejsce na wiezy", s["nr"], "zajete", s["fails"], "/", MAX_FAILS)
    if s["fails"] >= MAX_FAILS:
        s["closed"] = True                                                # free_slot ja pominie
        print("Wieza", s["nr"], "zamknieta - miejsce zajete", MAX_FAILS, "razy z rzedu")

def plan_place(quat, R_kd, p_kd, cloud, solver, target):
    """Plan odlozenia klocka, ktory zaraz chwycimy. Zwraca slownik planu albo None
    cloud - chmura kolizyjna ze zdjecia chwytania, target - punkt chwytu z choose_grasp."""
    i = free_slot()
    if i is None:
        print("Wieze pelne lub zamkniete - klocek do pudelka")
        return None
    s = slots[i]
    R_d = perception.quat_to_matrix(quat)                                 # dlon przy chwycie
    cands, b_new = place_candidates(s, R_d, R_kd, p_kd)                   # kandydaci od najtanszego

    all_cloud = np.vstack([cloud, field_cloud])                           # przeszkody z obu zdjec
    near = all_cloud[np.linalg.norm(all_cloud[:, :2] - b_new[:2], axis=1) < NEAR_XY]   # tylko punkty blisko wiezy
    near = near[near[:, 2] > s["top_z"] + SUPPORT_TOL]                    # bez gory wiezy i blatu zmierzonych z bledem
    bottom = b_new[2] - ITEM_SIZE / 2                                     # spod klocka przy puszczaniu

    work = all_cloud[all_cloud[:, 0] > TRANSIT_X_MIN]                     # scena bez podstawy robota
    scene_top = work[:, 2].max() if len(work) else TABLE_SURFACE_Z        # najwyzszy punkt sceny
    hang_grasp = hang(R_d, R_d @ R_kd, R_d @ p_kd)                        # zwis zaraz po chwycie

    reasons = [] if cands else ["pochylenie"]                             # brak kandydatow = za duze pochylenie dloni
    for cost, tilt, R_k_new, R_new, tcp_new in cands:
        held = finger_boxes(R_new, tcp_new, ITEM_SIZE / 2)                # palce zacisniete na klocku (zjazd)
        if lowest(held) < bottom + 0.002:
            reasons.append("palce pod klockiem")                          # zahaczylyby o gorny klocek wiezy
            continue
        fingers = finger_boxes(R_new, tcp_new, FINGER_OPEN)               # palce rozwarte (puszczanie i powrot)
        boxes = fingers + body_boxes(R_new, tcp_new) + [(b_new, R_k_new, np.full(3, ITEM_SIZE / 2))]
        hit = column_hit(near, boxes, [GRIP_MARGIN] * 4 + [BLOCK_MARGIN])
        if hit is not None:
            reasons.append("slup %.3f" % hit)                             # cos stoi w drodze (wysokosc przeszkody)
            continue
        hang_max = max(hang(R_new, R_k_new, R_new @ p_kd), hang_grasp)    # wiekszy zwis z chwytu i odlozenia
        transit = max(SAFE_Z, scene_top + TRANSIT_MARGIN + hang_max)      # wysokosc TCP przy przelocie
        q_new = to_quat(R_new)                                            # dlon przy odkladaniu jako kwaternion
        lift = np.array([target[0], target[1], transit])                  # koniec podnoszenia (nad punktem chwytu)
        above = np.array([tcp_new[0], tcp_new[1], transit])               # koniec przelotu (nad wieza)
        q_lift, ok = solve_ik(solver, lift, np.array(HOME_POSE[:7]), quat)   # ramie po podniesieniu klocka
        if not ok:
            reasons.append("ik przelot")                                  # nie podniesie klocka tak wysoko
            continue
        carry = kinematics.turn_path(solver, lift, above, q_lift, quat, q_new)   # przelot z obrotem dloni, jak w stanie carry
        if carry is None:
            reasons.append("ik obrot")                                    # po drodze brak rozwiazania albo przeskok przegubu
            continue
        down = kinematics.line_path(solver, above, tcp_new, carry[-1], q_new)    # zjazd pionowo, jak w stanie place
        if down is None or np.max(np.abs(down[-1][:6] - carry[-1][:6])) > perception.MAX_DESCENT_TURN:
            reasons.append("ik dol")                                      # zjazd wymaga przekrecenia ramienia
            continue
        s["fails"] = 0                                                    # krok 8: udany plan zeruje licznik
        print("Plan odlozenia: wieza", s["nr"], "warstwa", s["count"] + 1,
              "pochylenie dloni", round(np.degrees(tilt)), "palce", np.round(R_new[:, 1], 2),
              "TCP", np.round(tcp_new, 3), "kandydatow", len(cands), "odrzucone", reasons,
              "przelot", round(transit, 3), "(scena", round(scene_top, 3), "zwis", round(hang_max, 3), ")")
        return {"slot": i, "tcp": tcp_new, "quat": q_new, "b": b_new, "transit": transit}

    print("Brak pozy na wiezy", s["nr"], sorted(set(reasons)))
    check_blocked(s, near, b_new)                                         # krok 8: czy to wina miejsca, a nie chwytu
    return None

def finger_boxes(R, tcp, q):
    """Dwa palce jako prostopadlosciany: (srodek, osie w kolumnach, polowy wymiarow). q - rozwarcie palca od srodka."""
    y, z = R[:, 1], R[:, 2]                             # y - rozsuwanie palcow, z - os dloni w strone klocka
    tip = FINGER_Z + FINGER_LEN - TCP_OFFSET            # koniec palca 1.23 cm za TCP (w strone klocka)
    top = TCP_OFFSET - FINGER_Z                         # gora palca 4.16 cm przed TCP (w strone dloni)
    center = tcp - (top - tip) / 2 * z                  # srodek palca wzdluz osi dloni
    half = np.array([FINGER_WIDTH / 2, FINGER_THICK / 2, (top + tip) / 2])   # polowy wymiarow w osiach dloni
    return [(center + sg * (q + FINGER_THICK / 2) * y, R, half) for sg in (1.0, -1.0)]   # palec po obu stronach


def body_boxes(R, tcp):
    """Dlon i nadgarstek nad nia jako prostopadlosciany."""
    z = R[:, 2]
    palm0 = TCP_OFFSET - HAND_Z                         # spod dloni 3.4 cm nad TCP
    palm1 = palm0 + PALM_THICK                          # gora dloni 8 cm nad TCP
    palm = (tcp - (palm0 + palm1) / 2 * z, R, np.array([HAND_HALF_WIDTH, HAND_HALF_LEN, PALM_THICK / 2]))
    wrist = (tcp - (palm1 + WRIST_TOP) / 2 * z, R, np.array([WRIST_HALF, WRIST_HALF, (WRIST_TOP - palm1) / 2]))
    return [palm, wrist]


def lowest(boxes):
    # najnizszy punkt (z) sposrod naroznikow prostopadloscianow
    z = []
    for c, A, h in boxes:
        for s in itertools.product((-1.0, 1.0), repeat=3):
            z.append((c + A @ (h * np.array(s)))[2])
    return min(z)


def column_hit(cloud, boxes, margins):
    """Test wolnego slupa: bryly zjezdzaja pionowo z gory do tej pozy i wracaja ta sama droga.
    Punkt przeszkadza, gdy pionowa prosta przez niego przecina bryle, a on lezy nie nizej niz spod bryly.
    Zwraca wysokosc najwyzszej przeszkody albo None (slup wolny)."""
    worst = None
    for (c, A, h), m in zip(boxes, margins):
        hh = h + m                                                        # bryla powiekszona o zapas
        k = (cloud[:, 0:1] - c[0]) * A[0] + (cloud[:, 1:2] - c[1]) * A[1] - c[2] * A[2]   # (N, 3): punkt (x, y, 0) w osiach bryly
        lo = np.full(len(cloud), -np.inf)                                 # zakres wysokosci, na ktorym prosta jest w bryle
        hi = np.full(len(cloud), np.inf)
        inside = np.ones(len(cloud), dtype=bool)
        for i in range(3):
            az = A[2, i]                                                  # skladowa z osi i bryly
            if abs(az) > 1e-9:                                            # os nie jest pozioma: ogranicza wysokosc
                t1 = (-hh[i] - k[:, i]) / az
                t2 = (hh[i] - k[:, i]) / az
                lo = np.maximum(lo, np.minimum(t1, t2))
                hi = np.minimum(hi, np.maximum(t1, t2))
            else:                                                         # os pozioma: prosta w srodku albo obok
                inside &= np.abs(k[:, i]) <= hh[i]
        hit = inside & (lo <= hi) & (cloud[:, 2] >= lo)
        if np.any(hit):
            top = cloud[hit, 2].max()
            worst = top if worst is None else max(worst, top)
    return worst

def placed(plan):
    # klocek puszczony: zapamietaj, czego oczekujemy na nastepnym zdjeciu wiez
    global pending
    s = slots[plan["slot"]]
    pending = {"slot": plan["slot"], "xy": plan["b"][:2].copy(), "z": s["top_z"] + ITEM_SIZE, "count": s["count"] + 1}


def report_moved(before, after, plan):
    """Jak daleko przesunely sie inne klocki stojace na wiezach (prawda z symulatora, tylko do pomiarow)."""
    placed_i = np.argmin(np.linalg.norm(after - plan["b"], axis=1))     # odlozony klocek pomijamy
    moved = []
    for i in range(len(before)):
        if i == placed_i:
            continue
        if any(np.linalg.norm(before[i][:2] - s["xy"]) < ITEM_SIZE for s in slots):   # klocek stal na ktoryms polu
            moved.append(np.linalg.norm(after[i] - before[i]))
    if moved:
        print("Inne klocki na wiezach przesuniete max [mm]:", round(max(moved) * 1000, 1))


def summary():
    print("WIEZE:", ", ".join("%d: %d/%d%s" % (s["nr"], s["count"], scene.TOWER_HEIGHT, " zamknieta" if s["closed"] else "")
                              for s in slots))
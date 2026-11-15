"""Wyglad sceny do zdjec i filmu: czysta hala, cela robota jak w laboratorium, stol warsztatowy, klocki jak elementy.

Zmienia tylko wyglad. Kolizje, wymiary i polozenia stolu, klockow i robota zostaja te same,
wiec percepcja i ruch dzialaja tak samo jak w zwyklej scenie. Wszystko, co dokladamy (cela, szafa,
podloga celi), nie ma kolizji.
"""
import os
import numpy as np
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.experimental.materials import OmniGlassMaterial, OmniPbrMaterial
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd, UsdGeom, UsdLux

import scene

LOOK_ON = False                                         # True = ladna scena (wolniejsza symulacja, tylko do zdjec i filmu)

# hala
FACTORY_USD = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"   # hala; inne: warehouse.usd, warehouse_with_forklifts.usd
CLEAN_HALL = True                                       # True = z hali znikaja kartony, regaly, wozki; zostaja sciany, slupy, sufit, swiatla
FACTORY_YAW = None                                      # None = sam szuka miejsca w hali; liczba = obrot hali wokol pionu [deg]
FACTORY_POS = [0.0, 0.0, 0.0]                           # przesuniecie hali [m], uzywane razem z recznym FACTORY_YAW
SEARCH, SEARCH_STEP = 12.0, 1.5                         # gdzie szukac miejsca: do 12 m od srodka hali, co 1,5 m
NEAR = 15.0                                             # tlo filmu: liczy sie hala do tylu metrow od kamery, im blizej, tym wiecej

# cela robota (aluminiowe profile i oslony z poliweglanu), sam wyglad
CELL = True                                             # False = bez celi, szafy i podlogi celi
CELL_LO = (-0.75, -1.05)                                # rog celi [m] (x, y): za robotem i za rzedem wiez
CELL_HI = (1.55, 0.95)                                  # przeciwny rog; strona +y (od kamery filmowej) jest otwarta
CELL_H = 1.85                                           # wysokosc ramy [m]
PROFILE = 0.045                                         # przekroj profilu aluminiowego [m]
ZONE = 0.30                                             # podloga celi wystaje tyle poza rame [m]
CABINET = (-1.55, -1.45, -1.00, -1.05, 1.6)             # szafa sterownicza obok celi: x0, y0, x1, y1, wysokosc [m]

# obszar, w ktory nie moze wchodzic nic z hali (stanowisko, cela, szafa; kamera filmowa jest dokladana w dress)
CLEAR_LO = np.array([-1.75, -1.5, 0.02])
CLEAR_HI = np.array([1.9, 1.3, 2.5])

LAMP_POS = [0.4, 0.0, 2.4]                              # lampa nad stanowiskiem [m]
LAMP_SIZE = (1.2, 0.4)                                  # wymiary swiecacego prostokata [m]
LAMP_INTENSITY = 3000.0                                 # jasnosc lampy (za ciemno: x2, za jasno: /2)

# materialy: kolor jak na ekranie (sRGB), metalicznosc, chropowatosc
BLOCK_LOOKS = [                                         # klocki po kolei w tych materialach
    ((0.78, 0.79, 0.80), 1.0, 0.30),                    # aluminium surowe
    ((0.10, 0.28, 0.65), 0.7, 0.35),                    # aluminium anodowane na niebiesko
    ((0.85, 0.62, 0.25), 1.0, 0.28),                    # mosiadz
    ((0.32, 0.33, 0.35), 1.0, 0.45),                    # stal czerniona
]
EDGE_RADIUS = 0.002                                     # zaokraglenie krawedzi klockow tylko w obrazie [m], jak fazka po obrobce
TOP_LOOK = ((0.22, 0.30, 0.34), 0.0, 0.85)              # blat: szaroniebieska mata warsztatowa
FRAME_LOOK = ((0.72, 0.73, 0.71), 0.2, 0.50)            # rama stolu: jasnoszara stal malowana proszkowo
MARK_LOOK = ((0.95, 0.72, 0.05), 0.0, 0.60)             # miejsca na wieze: zolta tasma
WALL_LOOK = ((0.08, 0.09, 0.10), 0.0, 0.55)             # scianki pudelka: czarny plastik jak skrzynka magazynowa
RAMP_LOOK = ((0.55, 0.56, 0.58), 1.0, 0.40)             # rampy: stal
ALU_LOOK = ((0.80, 0.81, 0.82), 1.0, 0.35)              # profile celi: aluminium
ZONE_LOOK = ((0.62, 0.64, 0.66), 0.0, 0.35)             # podloga celi: jasnoszara zywica
TAPE_LOOK = ((0.95, 0.75, 0.05), 0.0, 0.60)             # obrys celi na podlodze: zolta tasma
CABINET_LOOK = ((0.80, 0.81, 0.78), 0.1, 0.45)          # szafa: jasnoszara (RAL 7035)
SCREEN_LOOK = ((0.03, 0.03, 0.04), 0.0, 0.08)           # panel operatora: ciemny ekran

LEG = 0.06                                              # bok profilu nog i ramy stolu [m]


def pbr(path, look, edge=0.0):
    """Material OmniPBR o stalym kolorze: look = (kolor sRGB, metalicznosc, chropowatosc)."""
    color, metallic, rough = look
    linear = [((c + 0.055) / 1.055) ** 2.4 for c in color]               # shader liczy w kolorze liniowym
    material = OmniPbrMaterial(path)
    material.set_input_values("diffuse_color_constant", [linear])
    material.set_input_values("metallic_constant", [metallic])
    material.set_input_values("reflection_roughness_constant", [rough])
    if edge > 0:
        material.set_input_values("round_edges_radius", [edge])          # zaokraglone krawedzie, sam obraz (bryla bez zmian)
    return material


def paint(paths, material):
    """Przypina material wizualny do primow (material fizyczny bez zmian)."""
    XformPrim(paths).apply_visual_materials(material)


def boxes(name, lows, highs):
    """Prostopadlosciany bez kolizji z rogow (x0, y0, z0) i (x1, y1, z1); zwraca ich sciezki."""
    lows, highs = np.array(lows, float), np.array(highs, float)
    cubes = Cube(paths=["/World/Look/{}{}".format(name, i) for i in range(len(lows))],
                 positions=((lows + highs) / 2).tolist(), scales=((highs - lows) / 2).tolist())
    return list(cubes.paths)


def table_frame():
    """Nogi, rama pod blatem i polka stolu."""
    cx, cy = scene.TABLE_CENTER
    hx, hy = scene.TABLE_HALF[0] - LEG, scene.TABLE_HALF[1] - LEG        # osie profili, wciete od krawedzi blatu
    under = scene.TABLE_SURFACE_Z - 0.05                                  # spod blatu (blat ma 5 cm)
    a = LEG / 2
    legs = [([x - a, y - a, 0.0], [x + a, y + a, under]) for x in (cx - hx, cx + hx) for y in (cy - hy, cy + hy)]
    rails = [([cx - hx, y - LEG / 3, under - 0.06], [cx + hx, y + LEG / 3, under]) for y in (cy - hy, cy + hy)]
    rails += [([x - LEG / 3, cy - hy, under - 0.06], [x + LEG / 3, cy + hy, under]) for x in (cx - hx, cx + hx)]
    shelf = [([cx - hx, cy - hy, 0.17], [cx + hx, cy + hy, 0.19])]
    parts = legs + rails + shelf
    return boxes("Table/Part", [p[0] for p in parts], [p[1] for p in parts])


def robot_cell():
    """Cela robota: rama z profili, oslony z poliweglanu, podloga celi z zoltym obrysem, szafa z panelem."""
    x0, y0 = CELL_LO
    x1, y1 = CELL_HI
    a, h = PROFILE / 2, CELL_H
    posts = [([x - a, y - a, 0.0], [x + a, y + a, h]) for x in (x0, x1) for y in (y0, y1)]
    posts += [([(x0 + x1) / 2 - a, y0 - a, 0.0], [(x0 + x1) / 2 + a, y0 + a, h])]          # srodkowy slupek z tylu
    rails = []
    for z in (0.10, h - a):                                               # rama dolna i gorna
        rails += [([x0, y0 - a, z - a], [x1, y0 + a, z + a])]             # tyl
        rails += [([x - a, y0, z - a], [x + a, y1, z + a]) for x in (x0, x1)]   # boki
    rails += [([x0, y1 - a, h - 2 * a], [x1, y1 + a, h])]                  # gora z przodu (przod otwarty)
    t = 0.003                                                             # polowa grubosci oslony [m]
    panels = [([x0 + a, y0 - t, 0.10 + a], [x1 - a, y0 + t, h - 2 * a])]                    # tyl
    panels += [([x - t, y0 + a, 0.10 + a], [x + t, y1 - a, h - 2 * a]) for x in (x0, x1)]   # boki
    zx0, zy0, zx1, zy1 = x0 - ZONE, y0 - ZONE, x1 + ZONE, y1 + ZONE
    zone = [([zx0, zy0, 0.0], [zx1, zy1, 0.002])]                         # podloga celi tuz nad podloga hali
    w = 0.05                                                              # szerokosc tasmy [m]
    tape = [([zx0, zy0, 0.0], [zx1, zy0 + w, 0.003]), ([zx0, zy1 - w, 0.0], [zx1, zy1, 0.003]),
            ([zx0, zy0, 0.0], [zx0 + w, zy1, 0.003]), ([zx1 - w, zy0, 0.0], [zx1, zy1, 0.003])]
    cx0, cy0, cx1, cy1, ch = CABINET
    cabinet = [([cx0, cy0, 0.0], [cx1, cy1, ch]), ([cx0, cy0, ch], [cx1, cy1, ch + 0.02])]  # szafa i daszek
    door = [((cx0 + cx1) / 2 - 0.004, cy1, 0.05), ((cx0 + cx1) / 2 + 0.004, cy1 + 0.004, ch - 0.05)]   # szczelina miedzy drzwiami
    handle = [(cx1 - 0.09, cy1, 0.85), (cx1 - 0.07, cy1 + 0.03, 1.05)]    # uchwyt
    screen = [(cx0 + 0.12, cy1, 1.25), (cx0 + 0.24, cy1 + 0.01, 1.42)]    # panel operatora na drzwiach
    out = {
        "alu": boxes("Cell/Profile", [p[0] for p in posts + rails] + [handle[0]], [p[1] for p in posts + rails] + [handle[1]]),
        "glass": boxes("Cell/Panel", [p[0] for p in panels], [p[1] for p in panels]),
        "zone": boxes("Cell/Zone", [p[0] for p in zone], [p[1] for p in zone]),
        "tape": boxes("Cell/Tape", [p[0] for p in tape], [p[1] for p in tape]),
        "cabinet": boxes("Cell/Cabinet", [p[0] for p in cabinet], [p[1] for p in cabinet]),
        "screen": boxes("Cell/Screen", [door[0], screen[0]], [door[1], screen[1]]),
    }
    return out


def objects(stage, root, size=4.0):
    """Rzeczy w hali jako prostopadlosciany: (sciezka, lo, hi, czy konstrukcja) i kawalki podlogi (lo, hi)."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    things, floors = [], []
    it = iter(Usd.PrimRange(stage.GetPrimAtPath(root)))
    for prim in it:
        if prim.HasAPI(UsdLux.LightAPI) or not prim.IsA(UsdGeom.Imageable):
            it.PruneChildren()                                            # swiatla hali zostaja
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            it.PruneChildren()
            continue
        lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
        if hi[2] < 0.3:                                                   # plaskie: podloga, linie na podlodze
            if np.max(hi[:2] - lo[:2]) > 2.0:
                floors.append((lo, hi))
            it.PruneChildren()
            continue
        parts = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Imageable)]
        if prim.GetPath() != root and (not parts or np.max(hi[:2] - lo[:2]) < size):
            things.append((prim.GetPath(), lo, hi, structure(lo, hi)))
            it.PruneChildren()
    return things, floors


def structure(lo, hi):
    """Czy to czesc budynku (zostaje), a nie rzecz w hali (karton, regal, wozek)."""
    size = hi - lo
    flat = np.sort(size[:2])                                              # (krotszy, dluzszy) bok w poziomie
    if lo[2] > 2.8:
        return True                                                       # sufit, belki, oprawy lamp
    if flat[1] > 12.0:
        return True                                                       # bardzo duza bryla: budynek, nie wyposazenie
    if size[2] >= 2.5 and flat[0] <= 0.6:
        return True                                                       # sciana albo jej kawalek (cienka i wysoka)
    if size[2] >= 3.0 and flat[1] <= 1.0:
        return True                                                       # slup
    return False


def placed(lo, hi, yaw, pos):
    """Prostopadlosciany po obrocie hali o yaw (co 90 st.) i przesunieciu o pos."""
    c, s = round(np.cos(np.radians(yaw))), round(np.sin(np.radians(yaw)))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    a, b = lo @ R.T + pos, hi @ R.T + pos
    return np.minimum(a, b), np.maximum(a, b)


def place_hall(things, floors, eye, target, hfov, aspect, lo_box, hi_box):
    """Obrot (co 90 st.) i przesuniecie hali: stanowisko na podlodze, z dala od scian, z hala blisko w tle filmu."""
    lo = np.array([t[1] for t in things]).reshape(-1, 3)
    hi = np.array([t[2] for t in things]).reshape(-1, 3)
    solid = np.array([t[3] for t in things], bool)                        # konstrukcja: nie do usuniecia
    flat = np.sort(hi[:, :2] - lo[:, :2], axis=1)
    shell = solid & (flat[:, 0] > 0.6) & (flat[:, 1] > 12.0)              # gruba i wielka bryla: skorupa budynku (pusta w srodku)
    stays = (solid if CLEAN_HALL else np.ones(len(things), bool)) & ~shell   # co zostanie w hali i moze byc tlem
    mid = (lo + hi) / 2
    along_x = hi[:, 0] - lo[:, 0] > hi[:, 1] - lo[:, 1]
    pts = []
    for f in np.linspace(0.1, 0.9, 5):                                    # 5 punktow wzdluz dluzszego boku kazdej rzeczy
        p = mid.copy()
        p[:, 0] = np.where(along_x, lo[:, 0] + f * (hi[:, 0] - lo[:, 0]), mid[:, 0])
        p[:, 1] = np.where(along_x, mid[:, 1], lo[:, 1] + f * (hi[:, 1] - lo[:, 1]))
        p[:, 2] = (lo[:, 2] + np.minimum(hi[:, 2], 2.5)) / 2
        pts.append(p)
    pts = np.stack(pts, axis=1).reshape(-1, 5, 3)
    flo = np.array([f[0] for f in floors]).reshape(-1, 3)
    fhi = np.array([f[1] for f in floors]).reshape(-1, 3)
    back = (eye - target) / np.linalg.norm(eye - target)                  # osie kamery filmowej
    right = np.cross([0.0, 0.0, 1.0], back)
    right = right / np.linalg.norm(right)
    up = np.cross(back, right)
    tan = np.tan(np.radians(hfov) / 2)
    corners = [np.array([x, y]) for x in (lo_box[0], hi_box[0]) for y in (lo_box[1], hi_box[1])]
    grid = np.arange(-SEARCH, SEARCH + 1e-6, SEARCH_STEP)
    best = None
    for yaw in (0, 90, 180, 270):
        rlo, rhi = placed(lo, hi, yaw, np.zeros(3))
        rflo, rfhi = placed(flo, fhi, yaw, np.zeros(3))
        c, s = round(np.cos(np.radians(yaw))), round(np.sin(np.radians(yaw)))
        rpts = pts @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        for dx in grid:
            for dy in grid:
                t = np.array([dx, dy, 0.0])
                hit = np.all(rhi + t >= lo_box, axis=1) & np.all(rlo + t <= hi_box, axis=1)   # wchodzi na stanowisko
                around = np.all(rlo[:, :2] + t[:2] <= lo_box[:2], axis=1) & np.all(rhi[:, :2] + t[:2] >= hi_box[:2], axis=1)
                if np.any(hit & solid & ~(shell & around)):
                    continue                                              # sciana albo slup na stanowisku (skorupa wokol jest ok)
                if len(flo) and not all(np.any(np.all((rflo[:, :2] + t[:2] <= q) & (rfhi[:, :2] + t[:2] >= q), axis=1))
                                        for q in corners):
                    continue                                              # stanowisko poza podloga hali
                d = rpts + t - eye
                depth = -(d @ back)
                seen = (depth > 0) & (np.abs(d @ right) < tan * depth) & (np.abs(d @ up) < tan / aspect * depth)
                dist = np.linalg.norm(d[..., :2], axis=-1)
                seen &= (dist > 3.0) & (stays & ~hit)[:, None]            # tlo, a nie cos przy kamerze albo usuwanego
                close = np.clip((NEAR - dist) / (NEAR - 3.0), 0.0, 1.0)  # 1 tuz za stanowiskiem, 0 od NEAR metrow
                view = (seen * close).sum() / 5
                cut = 0 if CLEAN_HALL else int(hit.sum())                 # przy czystej hali rzeczy i tak znikaja
                key = (view - 2 * cut, -np.hypot(dx, dy))
                if best is None or key > best[0]:
                    best = (key, yaw, [float(dx), float(dy), 0.0], view, cut)
    return best


def clean(stage, things, yaw, pos, lo_box, hi_box):
    """Wylacza z hali rzeczy: przy CLEAN_HALL wszystkie (bez konstrukcji), inaczej tylko te na stanowisku."""
    off, warn = [], []
    for path, lo, hi, solid in things:
        a, b = placed(lo, hi, yaw, pos)
        on_station = np.all(b >= lo_box) and np.all(a <= hi_box)
        around = np.all(a[:2] <= lo_box[:2]) and np.all(b[:2] >= hi_box[:2]) and np.min(hi[:2] - lo[:2]) > 0.6
        if solid:
            if on_station and not around:                                 # skorupa budynku wokol stanowiska jest ok
                warn.append(path)                                         # sciana na stanowisku: zostaje, ale ostrzegamy
        elif CLEAN_HALL or on_station:
            off.append(path)
    for path in off:
        stage.GetPrimAtPath(path).SetActive(False)                        # wylaczony: nie widac go i nie ma kolizji
    print("Hala: wylaczone rzeczy", len(off), "zostaje konstrukcji", sum(t[3] for t in things))
    if warn:
        print("UWAGA: przez stanowisko przechodzi", [p.name for p in warn][:5], "- zmien FACTORY_YAW / FACTORY_POS")


def save_map(things, floors, yaw, pos, eye, target, hfov, lo_box, hi_box, path):
    """Rysunek hali z gory: podloga szara, konstrukcja czerwona, usuniete rzeczy jasnoniebieskie, stanowisko i kadr zielone."""
    from PIL import Image, ImageDraw
    items = [placed(lo, hi, yaw, pos) + ("floor",) for lo, hi in floors]
    items += [placed(lo, hi, yaw, pos) + ("solid" if s else "thing",) for _, lo, hi, s in things]
    if not items:
        return
    lo_all = np.min([b[0] for b in items] + [lo_box], axis=0)
    hi_all = np.max([b[1] for b in items] + [hi_box], axis=0)
    k = 1400 / max(hi_all[0] - lo_all[0], hi_all[1] - lo_all[1])          # pikseli na metr
    w, h = int((hi_all[0] - lo_all[0]) * k) + 40, int((hi_all[1] - lo_all[1]) * k) + 40
    px = lambda x, y: (20 + (x - lo_all[0]) * k, h - 20 - (y - lo_all[1]) * k)   # os y w gore
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    style = {"floor": ("#dddddd", None), "thing": (None, "#9db8e8" if CLEAN_HALL else "#2a5fbf"), "solid": ("#d9534f", "#d9534f")}
    for kind in ("floor", "thing", "solid"):
        for lo, hi, kd in items:
            if kd == kind:
                (x0, y1), (x1, y0) = px(lo[0], lo[1]), px(hi[0], hi[1])
                draw.rectangle([x0, y0, x1, y1], fill=style[kind][0], outline=style[kind][1])
    (x0, y1), (x1, y0) = px(lo_box[0], lo_box[1]), px(hi_box[0], hi_box[1])
    draw.rectangle([x0, y0, x1, y1], outline="#2e7d4f", width=3)        # stanowisko z cela i kamera filmowa
    view = np.arctan2(target[1] - eye[1], target[0] - eye[0])
    for d in (-np.radians(hfov) / 2, np.radians(hfov) / 2):              # kadr kamery filmowej z gory, 15 m
        draw.line([px(eye[0], eye[1]), px(eye[0] + 15 * np.cos(view + d), eye[1] + 15 * np.sin(view + d))], fill="#2e7d4f", width=3)
    img.save(path)
    print("Hala: mapa z gory w", path)


def lamp(stage):
    """Prostokatna lampa nad stanowiskiem, swieci w dol."""
    light = UsdLux.RectLight.Define(stage, "/World/Look/Lamp")
    light.CreateWidthAttr(LAMP_SIZE[0])
    light.CreateHeightAttr(LAMP_SIZE[1])
    light.CreateIntensityAttr(LAMP_INTENSITY)
    UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*LAMP_POS))   # bez obrotu: swieci wzdluz -z, czyli w dol


def hall(stage, eye, target, hfov, aspect, lo_box, hi_box):
    """Wczytuje hale, ustawia ja wokol stanowiska i sprzata."""
    holder = UsdGeom.Xform.Define(stage, "/World/Hall")                   # uchwyt hali: przesuniecie i obrot wokol pionu
    move, spin = holder.AddTranslateOp(), holder.AddRotateZOp()
    move.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    spin.Set(0.0)
    stage_utils.add_reference_to_stage(usd_path=get_assets_root_path() + FACTORY_USD, path="/World/Hall/Factory")
    factory = stage.GetPrimAtPath("/World/Hall/Factory")
    if not (factory.IsValid() and factory.GetChildren()):
        print("Hala: nie udalo sie wczytac", FACTORY_USD, "- zostaje zwykla podloga")
        return
    things, floors = objects(stage, factory.GetPath())
    print("Hala: rzeczy", len(things), "w tym konstrukcji", sum(t[3] for t in things), "kawalkow podlogi", len(floors))
    yaw, pos = FACTORY_YAW, FACTORY_POS
    if yaw is None:                                                       # sam szuka miejsca na stanowisko w hali
        best = place_hall(things, floors, eye, target, hfov, aspect, lo_box, hi_box)
        if best is None:
            print("Hala: nie znalazlem wolnego miejsca - hala bez obrotu i przesuniecia")
            yaw, pos = 0, [0.0, 0.0, 0.0]
        else:
            yaw, pos = best[1], best[2]
            print("Hala: najlepsze miejsce -> FACTORY_YAW =", yaw, "FACTORY_POS =", pos, "hala w tle", round(best[3], 1))
    pos = np.array(pos, float)
    move.Set(Gf.Vec3d(*pos))
    spin.Set(float(yaw))
    clean(stage, things, yaw, pos, lo_box, hi_box)
    save_map(things, floors, yaw, pos, eye, target, hfov, lo_box, hi_box,
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "hala_mapa.png"))
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/ground_plane")).MakeInvisible()   # siatka znika, kolizja podlogi zostaje


def dress():
    """Ubiera zbudowana scene: hala, cela, stol warsztatowy, materialy, lampa. Wolac przed play()."""
    import film                                                           # kadr kamery filmowej (tu, bo film importuje scene)
    stage = stage_utils.get_current_stage()
    eye, target = np.array(film.FILM_EYE, float), np.array(film.FILM_TARGET, float)
    lo_box = np.minimum(CLEAR_LO, eye - 0.5)                              # stanowisko, cela i kamera filmowa z wszystkim miedzy nimi
    hi_box = np.maximum(CLEAR_HI, eye + 0.5)
    hall(stage, eye, target, film.FILM_HFOV, film.FILM_W / film.FILM_H, lo_box, hi_box)

    paint("/World/TableTop", pbr("/World/Look/Materials/Top", TOP_LOOK))
    paint(table_frame(), pbr("/World/Look/Materials/Frame", FRAME_LOOK))
    paint(scene.BOX_WALL_PATHS, pbr("/World/Look/Materials/Wall", WALL_LOOK))
    if scene.PLACE_MODE == "towers":
        paint(["/World/TowerMark{}".format(i) for i in range(scene.TOWER_COUNT)],
              pbr("/World/Look/Materials/Mark", MARK_LOOK))
    if scene.TEST_RAMPS:
        paint(["/World/Ramp{}".format(i) for i in range(len(scene.ramp_poses))],
              pbr("/World/Look/Materials/Ramp", RAMP_LOOK))
    for k, look in enumerate(BLOCK_LOOKS):
        paths = scene.item_paths[k::len(BLOCK_LOOKS)]                     # co n-ty klocek w tym samym materiale
        if paths:
            paint(paths, pbr("/World/Look/Materials/Block{}".format(k), look, EDGE_RADIUS))

    if CELL:
        cell = robot_cell()
        paint(cell["alu"], pbr("/World/Look/Materials/Alu", ALU_LOOK))
        paint(cell["glass"], OmniGlassMaterial("/World/Look/Materials/Glass"))   # przezroczysta oslona
        paint(cell["zone"], pbr("/World/Look/Materials/Zone", ZONE_LOOK))
        paint(cell["tape"], pbr("/World/Look/Materials/Tape", TAPE_LOOK))
        paint(cell["cabinet"], pbr("/World/Look/Materials/Cabinet", CABINET_LOOK))
        paint(cell["screen"], pbr("/World/Look/Materials/Screen", SCREEN_LOOK))

    lamp(stage)
    print("Wyglad: hala", FACTORY_USD, "czysta" if CLEAN_HALL else "z wyposazeniem", "cela" if CELL else "bez celi")

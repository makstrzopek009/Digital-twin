"""Wyglad sceny do zdjec i filmu: hala fabryczna, stol warsztatowy, klocki jak obrabiane elementy.

Zmienia tylko wyglad. Kolizje, wymiary i polozenia stolu, klockow i robota zostaja te same,
wiec percepcja i ruch dzialaja tak samo jak w zwyklej scenie.
"""
import os
import numpy as np
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.experimental.materials import OmniPbrMaterial
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd, UsdGeom, UsdLux

import scene

LOOK_ON = False                                         # True = ladna scena (wolniejsza symulacja, tylko do zdjec i filmu)
FACTORY_USD = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"   # hala; inne: warehouse.usd, warehouse_with_forklifts.usd
FACTORY_YAW = None                                      # None = sam szuka miejsca w hali; liczba = obrot hali wokol pionu [deg]
FACTORY_POS = [0.0, 0.0, 0.0]                           # przesuniecie hali [m], uzywane razem z recznym FACTORY_YAW
SEARCH, SEARCH_STEP = 12.0, 1.5                         # gdzie szukac miejsca: do 12 m od srodka hali, co 1,5 m
BIG = 4.0                                               # bryla dluzsza niz tyle [m] to sciana: nie wolno jej wycinac ze stanowiska
NEAR = 15.0                                             # tlo filmu: liczy sie hala do tylu metrow od kamery, im blizej, tym wiecej
CLEAR_LO = np.array([-0.8, -1.2, 0.02])                 # obszar stanowiska [m]: wszystko z hali, co tu wchodzi
CLEAR_HI = np.array([1.6, 1.2, 2.5])                    # (albo miedzy stanowisko a kamere filmowa), jest wylaczane; podloga zostaje
LAMP_POS = [0.4, 0.0, 2.4]                              # lampa nad stanowiskiem [m]
LAMP_SIZE = (1.2, 0.4)                                  # wymiary swiecacego prostokata [m]
LAMP_INTENSITY = 3000.0                                 # jasnosc lampy (za ciemno: x2, za jasno: /2)

BLOCK_LOOKS = [                                         # klocki po kolei w tych materialach: kolor rgb, metalicznosc, chropowatosc
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

LEG = 0.06                                              # bok profilu nog i ramy stolu [m]


def pbr(path, look, edge=0.0):
    """Material OmniPBR o stalym kolorze: look = (kolor rgb, metalicznosc, chropowatosc)."""
    color, metallic, rough = look
    linear = [((c + 0.055) / 1.055) ** 2.4 for c in color]               # kolor podany jak na ekranie (sRGB), shader liczy w liniowym
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


def table_frame():
    """Nogi, rama pod blatem i polka - sam wyglad, bez kolizji."""
    cx, cy = scene.TABLE_CENTER
    hx, hy = scene.TABLE_HALF[0] - LEG, scene.TABLE_HALF[1] - LEG        # osie profili, wciete od krawedzi blatu
    under = scene.TABLE_SURFACE_Z - 0.05                                  # spod blatu (blat ma 5 cm)
    corners = [[cx + sx * hx, cy + sy * hy] for sx in (-1, 1) for sy in (-1, 1)]
    legs = Cube(paths=["/World/Look/Table/Leg{}".format(i) for i in range(4)],
                positions=[[x, y, under / 2] for x, y in corners],
                scales=[LEG / 2, LEG / 2, under / 2])                     # nogi od podlogi do spodu blatu
    rails = Cube(paths=["/World/Look/Table/Rail{}".format(i) for i in range(4)],
                 positions=[[cx, cy - hy, under - 0.03], [cx, cy + hy, under - 0.03],
                            [cx - hx, cy, under - 0.03], [cx + hx, cy, under - 0.03]],
                 scales=[[hx, LEG / 3, 0.03], [hx, LEG / 3, 0.03],
                         [LEG / 3, hy, 0.03], [LEG / 3, hy, 0.03]])       # rama tuz pod blatem
    shelf = Cube(paths="/World/Look/Table/Shelf",
                 positions=[cx, cy, 0.18], scales=[hx, hy, 0.01])         # polka nisko nad podloga
    return list(legs.paths) + list(rails.paths) + list(shelf.paths)


def objects(stage, root, size=4.0):
    """Rzeczy w hali jako prostopadlosciany: rzeczy stojace (sciezka, lo, hi) i kawalki podlogi (lo, hi)."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    things, floors = [], []
    it = iter(Usd.PrimRange(stage.GetPrimAtPath(root)))
    for prim in it:
        if prim.HasAPI(UsdLux.LightAPI) or not prim.IsA(UsdGeom.Imageable):
            it.PruneChildren()                                            # swiatla hali pomijamy
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            it.PruneChildren()
            continue
        lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
        if hi[2] < 0.3:                                                   # plaskie: podloga, linie na podlodze
            if np.max(hi[:2] - lo[:2]) > 2.0:
                floors.append((lo, hi))                                   # kawalek podlogi, na ktorym moze stac stanowisko
            it.PruneChildren()
            continue
        parts = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Imageable)]
        if prim.GetPath() != root and (not parts or np.max(hi[:2] - lo[:2]) < size):
            things.append((prim.GetPath(), lo, hi))                       # regal, wozek, skrzynia albo pojedyncza bryla (np. sciana)
            it.PruneChildren()
    return things, floors


def place_hall(things, floors, eye, target, hfov, aspect, lo_box, hi_box):
    """Obrot hali (co 90 st.) i przesuniecie: stanowisko na wolnej podlodze z dala od scian, jak najwiecej rzeczy w tle filmu."""
    if not things:
        return (0, 0), 0, [0.0, 0.0, 0.0], 0, 0
    lo = np.array([t[1] for t in things])
    hi = np.array([t[2] for t in things])
    big = np.max(hi[:, :2] - lo[:, :2], axis=1) > BIG                    # sciany i inne duze bryly: nie wolno ich wycinac
    mid = (lo + hi) / 2
    along_x = hi[:, 0] - lo[:, 0] > hi[:, 1] - lo[:, 1]                   # dluzszy bok w poziomie
    pts = []
    for f in np.linspace(0.1, 0.9, 5):                                    # 5 punktow wzdluz dluzszego boku kazdej rzeczy
        p = mid.copy()
        p[:, 0] = np.where(along_x, lo[:, 0] + f * (hi[:, 0] - lo[:, 0]), mid[:, 0])
        p[:, 1] = np.where(along_x, mid[:, 1], lo[:, 1] + f * (hi[:, 1] - lo[:, 1]))
        p[:, 2] = (lo[:, 2] + np.minimum(hi[:, 2], 2.5)) / 2              # srodek wysokosci (gora regalu i tak zwykle poza kadrem)
        pts.append(p)
    pts = np.stack(pts, axis=1)                                           # (rzeczy, 5, xyz)
    flo = np.array([f[0] for f in floors]).reshape(-1, 3)
    fhi = np.array([f[1] for f in floors]).reshape(-1, 3)
    back = (eye - target) / np.linalg.norm(eye - target)                  # osie kamery filmowej
    right = np.cross([0.0, 0.0, 1.0], back)
    right = right / np.linalg.norm(right)
    up = np.cross(back, right)
    tan = np.tan(np.radians(hfov) / 2)
    corners = [np.array([x, y]) for x in (lo_box[0], hi_box[0]) for y in (lo_box[1], hi_box[1])]   # rogi obszaru stanowiska
    grid = np.arange(-SEARCH, SEARCH + 1e-6, SEARCH_STEP)
    best = None
    for yaw in (0, 90, 180, 270):
        c, s = round(np.cos(np.radians(yaw))), round(np.sin(np.radians(yaw)))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        a, b = lo @ R.T, hi @ R.T
        rlo, rhi = np.minimum(a, b), np.maximum(a, b)                     # prostopadlosciany po obrocie (dokladne co 90 st.)
        rpts = pts @ R.T
        fa, fb = flo @ R.T, fhi @ R.T
        rflo, rfhi = np.minimum(fa, fb), np.maximum(fa, fb)
        for dx in grid:
            for dy in grid:
                t = np.array([dx, dy, 0.0])
                hit = np.all(rhi + t >= lo_box, axis=1) & np.all(rlo + t <= hi_box, axis=1)   # wchodzi na stanowisko
                if np.any(hit & big):
                    continue                                              # sciana przez stanowisko albo kamere
                if len(flo) and not all(np.any(np.all((rflo[:, :2] + t[:2] <= q) & (rfhi[:, :2] + t[:2] >= q), axis=1))
                                        for q in corners):
                    continue                                              # stanowisko poza podloga hali
                d = rpts + t - eye
                depth = -(d @ back)
                seen = (depth > 0) & (np.abs(d @ right) < tan * depth) & (np.abs(d @ up) < tan / aspect * depth)
                dist = np.linalg.norm(d[..., :2], axis=-1)
                seen &= (dist > 3.0) & ~hit[:, None] & ~big[:, None]      # tlo, a nie cos tuz przy kamerze albo na stanowisku
                close = np.clip((NEAR - dist) / (NEAR - 3.0), 0.0, 1.0)  # 1 tuz za stanowiskiem, 0 od NEAR metrow
                view = (seen * close).sum() / 5
                score = view - 2 * hit.sum()                              # bliskie rzeczy w tle minus rzeczy wyciete ze stanowiska
                key = (score, -np.hypot(dx, dy))                          # przy remisie blizej srodka hali
                if best is None or key > best[0]:
                    best = (key, yaw, [float(dx), float(dy), 0.0], view, int(hit.sum()))
    return best


def save_map(things, floors, yaw, pos, eye, target, hfov, lo_box, hi_box, path):
    """Rysunek hali z gory po ustawieniu: podloga szara, rzeczy niebieskie, sciany czerwone, stanowisko i kadr kamery zielone."""
    from PIL import Image, ImageDraw
    c, s = round(np.cos(np.radians(yaw))), round(np.sin(np.radians(yaw)))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    def place(lo, hi):                                                    # prostopadloscian po obrocie i przesunieciu hali
        a, b = lo @ R.T + pos, hi @ R.T + pos
        return np.minimum(a, b), np.maximum(a, b)
    boxes = [place(lo, hi) + ("floor",) for lo, hi in floors]
    boxes += [place(lo, hi) + ("big" if np.max(hi[:2] - lo[:2]) > BIG else "thing",) for _, lo, hi in things]
    if not boxes:
        return
    lo_all = np.min([b[0] for b in boxes] + [lo_box], axis=0)
    hi_all = np.max([b[1] for b in boxes] + [hi_box], axis=0)
    k = 1400 / max(hi_all[0] - lo_all[0], hi_all[1] - lo_all[1])          # pikseli na metr
    w, h = int((hi_all[0] - lo_all[0]) * k) + 40, int((hi_all[1] - lo_all[1]) * k) + 40
    px = lambda x, y: (20 + (x - lo_all[0]) * k, h - 20 - (y - lo_all[1]) * k)   # os y w gore
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    style = {"floor": ("#dddddd", None), "thing": (None, "#2a5fbf"), "big": ("#d9534f", "#d9534f")}
    for kind in ("floor", "thing", "big"):
        for lo, hi, kd in boxes:
            if kd == kind:
                (x0, y1), (x1, y0) = px(lo[0], lo[1]), px(hi[0], hi[1])
                draw.rectangle([x0, y0, x1, y1], fill=style[kind][0], outline=style[kind][1])
    (x0, y1), (x1, y0) = px(lo_box[0], lo_box[1]), px(hi_box[0], hi_box[1])
    draw.rectangle([x0, y0, x1, y1], outline="#2e7d4f", width=3)        # stanowisko z kamera filmowa
    view = np.arctan2(target[1] - eye[1], target[0] - eye[0])
    for d in (-np.radians(hfov) / 2, np.radians(hfov) / 2):              # kadr kamery filmowej z gory, 15 m
        draw.line([px(eye[0], eye[1]), px(eye[0] + 15 * np.cos(view + d), eye[1] + 15 * np.sin(view + d))], fill="#2e7d4f", width=3)
    img.save(path)
    print("Hala: mapa z gory w", path)


def clear_station(stage, root, lo_box, hi_box):
    """Wylacza z hali regaly i inne rzeczy, ktore wchodza na stanowisko albo zaslaniaja kamere filmowa."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    hits, kept = [], []
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
        if np.any(hi < lo_box) or np.any(lo > hi_box):
            it.PruneChildren()                                            # poza stanowiskiem, w srodku tez nic nie wchodzi
            continue
        parts = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Imageable)]
        if prim.GetPath() != root and (not parts or np.max(hi[:2] - lo[:2]) < 4.0):
            if np.max(hi[:2] - lo[:2]) > BIG:
                kept.append(prim.GetPath())                               # sciana: zostaje, wycieta zrobilaby dziure w hali
            else:
                hits.append(prim.GetPath())                               # maly obiekt (regal, wozek)
            it.PruneChildren()
    for path in hits:
        stage.GetPrimAtPath(path).SetActive(False)                        # wylaczony: nie widac go i nie ma kolizji
    print("Hala: wylaczone ze stanowiska", len(hits), [p.name for p in hits][:10])
    if kept:
        print("UWAGA: przez stanowisko przechodzi", [p.name for p in kept][:5], "- zmien FACTORY_YAW / FACTORY_POS")


def lamp(stage):
    """Prostokatna lampa nad stanowiskiem, swieci w dol."""
    light = UsdLux.RectLight.Define(stage, "/World/Look/Lamp")
    light.CreateWidthAttr(LAMP_SIZE[0])
    light.CreateHeightAttr(LAMP_SIZE[1])
    light.CreateIntensityAttr(LAMP_INTENSITY)
    UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*LAMP_POS))   # bez obrotu: swieci wzdluz -z, czyli w dol


def dress():
    """Ubiera zbudowana scene: hala, stol warsztatowy, materialy, lampa. Wolac przed play()."""
    import film                                                           # kadr kamery filmowej (tu, bo film importuje scene)
    stage = stage_utils.get_current_stage()
    eye, target = np.array(film.FILM_EYE, float), np.array(film.FILM_TARGET, float)
    aspect = film.FILM_W / film.FILM_H
    lo_box = np.minimum(CLEAR_LO, eye - 0.5)                              # stanowisko razem z kamera filmowa i wszystkim miedzy nimi
    hi_box = np.maximum(CLEAR_HI, eye + 0.5)

    hall = UsdGeom.Xform.Define(stage, "/World/Hall")                     # uchwyt hali: przesuniecie i obrot wokol pionu
    move, spin = hall.AddTranslateOp(), hall.AddRotateZOp()
    move.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    spin.Set(0.0)
    stage_utils.add_reference_to_stage(usd_path=get_assets_root_path() + FACTORY_USD, path="/World/Hall/Factory")
    factory = stage.GetPrimAtPath("/World/Hall/Factory")
    if factory.IsValid() and factory.GetChildren():
        yaw, pos = FACTORY_YAW, FACTORY_POS
        things, floors = objects(stage, factory.GetPath())
        lo_all = np.min([t[1] for t in things] + [f[0] for f in floors], axis=0) if things or floors else np.zeros(3)
        hi_all = np.max([t[2] for t in things] + [f[1] for f in floors], axis=0) if things or floors else np.zeros(3)
        print("Hala: zakres x", np.round([lo_all[0], hi_all[0]], 1), "y", np.round([lo_all[1], hi_all[1]], 1),
              "z", np.round([lo_all[2], hi_all[2]], 1), "rzeczy", len(things), "w tym scian", sum(np.max(t[2][:2] - t[1][:2]) > BIG for t in things),
              "kawalkow podlogi", len(floors))
        if yaw is None:                                                   # sam szuka miejsca na stanowisko w hali
            best = place_hall(things, floors, eye, target, film.FILM_HFOV, aspect, lo_box, hi_box)
            if best is None:
                print("Hala: nie znalazlem wolnego miejsca - hala bez obrotu i przesuniecia")
                yaw, pos = 0, [0.0, 0.0, 0.0]
            else:
                yaw, pos = best[1], best[2]
                print("Hala: najlepsze miejsce -> FACTORY_YAW =", yaw, "FACTORY_POS =", pos,
                      "w tle", round(best[3], 1), "do wyciecia", best[4])
        move.Set(Gf.Vec3d(*pos))
        spin.Set(float(yaw))
        save_map(things, floors, yaw, np.array(pos, float), eye, target, film.FILM_HFOV, lo_box, hi_box,
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "hala_mapa.png"))
        clear_station(stage, factory.GetPath(), lo_box, hi_box)
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/ground_plane")).MakeInvisible()   # siatka znika, kolizja podlogi zostaje
    else:
        print("Hala: nie udalo sie wczytac", FACTORY_USD, "- zostaje zwykla podloga")

    frame = table_frame()
    paint("/World/TableTop", pbr("/World/Look/Materials/Top", TOP_LOOK))
    paint(frame, pbr("/World/Look/Materials/Frame", FRAME_LOOK))
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

    lamp(stage)
    print("Wyglad: hala", FACTORY_USD, "klockow", len(scene.item_paths), "materialow klockow", len(BLOCK_LOOKS))

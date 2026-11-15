"""Wyglad sceny do zdjec i filmu: hala fabryczna, stol warsztatowy, klocki jak obrabiane elementy.

Zmienia tylko wyglad. Kolizje, wymiary i polozenia stolu, klockow i robota zostaja te same,
wiec percepcja i ruch dzialaja tak samo jak w zwyklej scenie.
"""
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
FACTORY_POS = [0.0, 0.0, 0.0]                           # przesuniecie hali [m], gdy stanowisko wypada w zlym miejscu
FACTORY_YAW = None                                      # obrot hali wokol pionu [deg]; None = sam wybiera 0/90/180/270, zeby regaly byly w tle filmu
CLEAR_LO = np.array([-0.8, -1.2, 0.02])                 # obszar stanowiska [m]: wszystko z hali, co tu wchodzi
CLEAR_HI = np.array([1.6, 1.2, 2.5])                    # (albo miedzy stanowisko a kamere filmowa), jest wylaczane; podloga zostaje
LAMP_POS = [0.4, 0.0, 2.4]                              # lampa nad stanowiskiem [m]
LAMP_SIZE = (1.2, 0.4)                                  # wymiary swiecacego prostokata [m]
LAMP_INTENSITY = 8000.0                                 # jasnosc lampy (za ciemno: x2, za jasno: /2)

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
    material = OmniPbrMaterial(path)
    material.set_input_values("diffuse_color_constant", [list(color)])
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
    """Rzeczy w hali (regaly, wozki, skrzynie) jako prostopadlosciany (lo, hi); duze grupy rozbija na czesci."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    found = []
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
        if hi[2] < 0.3:                                                   # podloga, linie na podlodze, plaskie rzeczy
            it.PruneChildren()
            continue
        parts = [c for c in prim.GetChildren() if c.IsA(UsdGeom.Imageable)]
        if prim.GetPath() != root and (not parts or np.max(hi[:2] - lo[:2]) < size):
            found.append((prim.GetPath(), lo, hi))                        # maly obiekt albo pojedyncza bryla
            it.PruneChildren()
    return found


def in_frame(p, eye, target, hfov, aspect):
    """Czy punkt p jest w kadrze kamery patrzacej z eye na target."""
    back = (eye - target) / np.linalg.norm(eye - target)                  # os z kamery (kamera patrzy wzdluz -z)
    right = np.cross([0.0, 0.0, 1.0], back)
    right = right / np.linalg.norm(right)
    up = np.cross(back, right)
    d = p - eye
    depth = -d @ back                                                     # odleglosc przed kamera
    t = np.tan(np.radians(hfov) / 2)
    return depth > 0 and abs(d @ right) < t * depth and abs(d @ up) < t / aspect * depth


def turn(p, yaw):
    """Punkt p po obrocie hali o yaw [deg] wokol pionu i przesunieciu o FACTORY_POS."""
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]) + FACTORY_POS


def background(found, yaw, eye, target, hfov, aspect, lo_box, hi_box):
    """Ile rzeczy z hali bedzie w tle filmu przy danym obrocie hali (czesci widocznych punktow)."""
    score = 0.0
    for path, lo, hi in found:
        z = (lo[2] + min(hi[2], 2.5)) / 2                                 # srodek wysokosci (gora regalu i tak zwykle poza kadrem)
        long = 0 if hi[0] - lo[0] > hi[1] - lo[1] else 1                  # dluzszy bok w poziomie
        for f in np.linspace(0.1, 0.9, 5):                                # 5 punktow wzdluz dluzszego boku
            p = (lo + hi) / 2
            p[long] = lo[long] + f * (hi[long] - lo[long])
            p[2] = z
            q = turn(p, yaw)
            inside = np.all(q >= lo_box) and np.all(q <= hi_box)          # i tak zostanie wylaczony ze stanowiska
            far = np.linalg.norm(q[:2] - eye[:2]) > 3.0                   # tlo, a nie cos tuz przy kamerze
            if far and not inside and in_frame(q, eye, target, hfov, aspect):
                score += 0.2
    return score


def clear_station(stage, root, lo_box, hi_box):
    """Wylacza z hali regaly i inne rzeczy, ktore wchodza na stanowisko albo zaslaniaja kamere filmowa."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    hits = []
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
            hits.append(prim.GetPath())                                   # maly obiekt (regal, wozek) albo pojedyncza bryla
            it.PruneChildren()
    for path in hits:
        stage.GetPrimAtPath(path).SetActive(False)                        # wylaczony: nie widac go i nie ma kolizji
    print("Hala: wylaczone ze stanowiska", len(hits), [p.name for p in hits][:10])


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
        yaw = FACTORY_YAW
        if yaw is None:                                                   # obrot, przy ktorym najwiecej rzeczy z hali jest w tle filmu
            found = objects(stage, factory.GetPath())
            scores = {a: background(found, a, eye, target, film.FILM_HFOV, aspect, lo_box, hi_box) for a in (0, 90, 180, 270)}
            yaw = max(scores, key=scores.get)
            print("Hala: rzeczy w tle przy obrocie 0/90/180/270:", [round(scores[a], 1) for a in (0, 90, 180, 270)], "-> obrot", yaw)
        move.Set(Gf.Vec3d(*FACTORY_POS))
        spin.Set(float(yaw))
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

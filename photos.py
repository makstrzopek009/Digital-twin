"""Zdjecia do pracy: pudelko ze sciankami i bez, cale stanowisko, obraz z kamery robota (kolor i glebia)."""
import os
import time
import numpy as np
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

import scene
import perception
import film

PHOTOS_ON = True                                # False = bez zdjec do pracy
BOX_PATH = "/World/BoxCamera"                   # kamera do zblizenia na pudelko
BOX_EYE = np.array([1.53, 0.28, 1.55])          # skad patrzy [m]: przed pudelkiem, troche z boku i z gory
BOX_TARGET = np.array([0.50, 0.00, 0.80])       # srodek pudelka
BOX_HFOV = 45.0                                 # poziomy kat widzenia [deg]
STATION_PATH = "/World/StationCamera"           # kamera stanowiska, gdy film jest wylaczony (kadr jak filmowa)
DEPTH_RANGE = (0.40, 0.60)                      # zakres glebi na zdjeciu glebi [m]: od gory stosu do blatu

box = None                                      # obraz z kamery pudelka
station = None                                  # obraz z kamery stanowiska
folder = None                                   # folder na zdjecia
taken = set()                                   # zrobione zdjecia (kazde tylko raz)


def camera(path, eye, target, hfov):
    """Nieruchoma kamera w scenie, obraz kolorowy w rozdzielczosci filmu."""
    cam = RtxCamera(path=path, translations=eye, orientations=np.array(film.look_at(eye, target)), tick_rate=30.0)
    scene.configure_camera_optics(path, hfov_deg=hfov, far=1000.0)                    # zasieg do 1 km, zeby bylo widac tlo
    return CameraSensor(cam, resolution=(film.FILM_H, film.FILM_W), annotators=["rgb"])


def setup():
    """Kamery do zdjec i folder praca_<data>_<godzina> obok main.py."""
    global box, station, folder
    if not PHOTOS_ON:
        return
    box = camera(BOX_PATH, BOX_EYE, BOX_TARGET, BOX_HFOV)
    station = film.sensor                                                             # kamera filmowa, jesli jest
    if station is None:
        station = camera(STATION_PATH, film.FILM_EYE, film.FILM_TARGET, film.FILM_HFOV)   # film wylaczony: osobna kamera w tym samym miejscu
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), time.strftime("praca_%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    print("Zdjecia do pracy:", folder)


def save(name, img):
    """Zapis obrazu jako png."""
    from PIL import Image
    Image.fromarray(img).save(os.path.join(folder, name + ".png"))
    taken.add(name)
    print("Zdjecie:", name)


def snap(name, sensor):
    """Zdjecie kolorowe z kamery; kazda nazwa tylko raz."""
    if folder is None or sensor is None or name in taken:
        return
    data, info = sensor.get_data("rgb")
    if data is None:                                                                  # obraz jeszcze nie gotowy
        return
    save(name, perception.to_numpy(data)[..., :3].astype(np.uint8))                   # rgba -> rgb


def depth_image(depth):
    """Glebia w kolorach (paleta viridis): blisko = zolty, daleko = fiolet, brak pomiaru = czarny."""
    d = np.squeeze(np.asarray(depth, dtype=float))                                    # (wysokosc, szerokosc) w metrach
    ok = np.isfinite(d) & (d > 0)
    t = np.clip((DEPTH_RANGE[1] - np.where(ok, d, DEPTH_RANGE[1])) / (DEPTH_RANGE[1] - DEPTH_RANGE[0]), 0.0, 1.0)   # 1 = blisko
    stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=float)   # kolory viridis
    x = t * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)                           # miedzy ktorymi kolorami
    f = (x - i)[..., None]
    img = stops[i] * (1 - f) + stops[i + 1] * f                                       # plynne przejscie miedzy kolorami
    img[~ok] = 0
    return img.astype(np.uint8)


def before_walls():
    """Przed usunieciem scianek: pudelko ze sciankami."""
    snap("1_pudelko_ze_sciankami", box)


def after_walls():
    """Po usunieciu scianek i ulozeniu sie klockow: pudelko bez scianek i cale stanowisko."""
    snap("2_pudelko_bez_scianek", box)
    snap("3_stanowisko", station)


def robot_view(sensor, depth):
    """Pierwsze zdjecie stosu z kamery robota: obraz kolorowy i glebia."""
    if folder is None or "5_kamera_robota_glebia" in taken:
        return
    snap("4_kamera_robota_kolor", sensor)                                             # wymaga ROBOT_RGB = True w scene.py
    save("5_kamera_robota_glebia", depth_image(depth))


def end():
    """Koniec przebiegu: stanowisko z wiezami."""
    snap("6_stanowisko_koniec", station)

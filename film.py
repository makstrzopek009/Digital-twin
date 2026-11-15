"""Nagrywanie filmu z osobnej kamery w scenie, klatka po klatce (tempo symulacji, nie komputera)."""
import os
import time
import numpy as np
import omni.kit.viewport.utility as vp_utils
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

import scene
import perception

FILM_ON = True                                  # False = bez nagrywania (symulacja szybsza)
FILM_PATH = "/World/FilmCamera"                 # sciezka kamery filmowej
FILM_EYE = np.array([1.39, 2.58, 1.45])         # skad patrzy [m]: z drugiej strony blatu, nisko, zeby za stanowiskiem bylo widac hale
FILM_TARGET = np.array([0.30, -0.08, 1.08])     # na co patrzy: stos z przodu, wieze za nim, cale ramie robota
FILM_HFOV = 50.0                                # poziomy kat widzenia [deg]
FILM_W, FILM_H = 1920, 1080                     # rozdzielczosc filmu
FILM_FPS = 30                                   # klatki na sekunde w pliku
FILM_EVERY = 2                                  # co ile krokow fizyki (60 Hz) klatka: 1 = zwolnione x2, 2 = tempo rzeczywiste, 4 = przyspieszone x2
PHOTO_EVERY = 0                                 # co ile krokow fizyki zdjecie png z tej samej kamery: 0 = bez zdjec, 300 = co 5 s

sensor = None                                   # obraz z kamery filmowej
writer = None                                   # plik mp4
frames_dir = None                               # folder na png, gdy nie ma imageio
photos_dir = None                               # folder na zdjecia
photos = 0                                      # ile zdjec zapisano
calls = 0                                       # ile razy wywolano grab
saved = 0                                       # ile klatek zapisano


def look_at(eye, target):
    """Obrot kamery USD (w, x, y, z): patrzy z eye na target, gora obrazu do gory."""
    back = (eye - target) / np.linalg.norm(eye - target)    # os z kamery: od celu do kamery (kamera patrzy wzdluz -z)
    right = np.cross([0.0, 0.0, 1.0], back)                 # os x kamery: poziomo w prawo
    right = right / np.linalg.norm(right)                   # dlugosc 1
    up = np.cross(back, right)                              # os y kamery: gora obrazu
    return scene.wxyz(np.column_stack([right, up, back]))   # kolumny = osie kamery w swiecie


def setup():
    """Kamera filmowa w scenie, plik filmu i folder na zdjecia."""
    global sensor, writer, frames_dir, photos_dir
    if not FILM_ON and not PHOTO_EVERY:
        return
    cam = RtxCamera(path=FILM_PATH, translations=FILM_EYE,
                    orientations=np.array(look_at(FILM_EYE, FILM_TARGET)), tick_rate=60.0 / FILM_EVERY)   # stala kamera obok stolu, render tylko potrzebnych klatek
    scene.configure_camera_optics(FILM_PATH, hfov_deg=FILM_HFOV)                              # kat widzenia i zasieg jak przy D435i
    sensor = CameraSensor(cam, resolution=(FILM_H, FILM_W), annotators=["rgb"])                # obraz kolorowy (wysokosc, szerokosc)
    vp = vp_utils.get_active_viewport()                                                       # glowny widok (bez okna: None)
    if vp is not None:
        vp.set_active_camera(FILM_PATH)                                                       # okno pokazuje to, co nagrywamy
    name = time.strftime("film_%Y%m%d_%H%M%S")                                                # nazwa pliku z data i godzina
    folder = os.path.dirname(os.path.abspath(__file__))                                       # zapis obok main.py
    if PHOTO_EVERY:
        photos_dir = os.path.join(folder, name.replace("film_", "zdjecia_"))                  # np. zdjecia_20261115_201500
        os.makedirs(photos_dir, exist_ok=True)
        print("Zdjecia: co", PHOTO_EVERY, "krokow do", photos_dir)
    if not FILM_ON:
        return
    try:
        import imageio.v2 as imageio
        path = os.path.join(folder, name + ".mp4")
        writer = imageio.get_writer(path, fps=FILM_FPS, codec="libx264", quality=8, macro_block_size=8,
                                    ffmpeg_params=["-preset", "veryfast"])                    # 1080 dzieli sie przez 8, szybkie kodowanie
        print("Film: nagrywam do", path)
    except Exception as e:                                                                    # brak imageio albo ffmpeg
        frames_dir = os.path.join(folder, name)
        os.makedirs(frames_dir, exist_ok=True)
        print("Film: brak imageio (", e, ") - klatki png do", frames_dir)


def grab():
    """Co FILM_EVERY krokow fizyki dopisuje klatke do filmu, co PHOTO_EVERY zapisuje zdjecie."""
    global calls, saved, photos
    film = writer is not None or frames_dir is not None     # film jeszcze sie nagrywa
    if not film and photos_dir is None:                     # nic nie zapisujemy albo juz zamkniete
        return
    calls += 1
    film_now = film and calls % FILM_EVERY == 0             # ten krok trafia do filmu
    photo_now = photos_dir is not None and calls % PHOTO_EVERY == 0   # w tym kroku zdjecie
    if not film_now and not photo_now:
        return
    rgb, info = sensor.get_data("rgb")                      # obraz z kamery filmowej
    if rgb is None:                                         # pierwsze klatki: obraz jeszcze nie gotowy
        return
    img = perception.to_numpy(rgb)[..., :3].astype(np.uint8)   # rgba -> rgb
    if photo_now:
        from PIL import Image
        Image.fromarray(img).save(os.path.join(photos_dir, "%03d_%04ds.png" % (photos, calls // 60)))   # numer i sekunda symulacji
        photos += 1
    if not film_now:
        return
    if writer is not None:
        writer.append_data(img)                             # klatka prosto do mp4
    else:
        from PIL import Image                               # zapas: klatka jako png
        Image.fromarray(img).save(os.path.join(frames_dir, "%05d.png" % saved), compress_level=1)
    saved += 1


def close():
    """Domyka plik filmu (bez tego mp4 sie nie otworzy) i podsumowuje zdjecia."""
    global writer, frames_dir, photos_dir
    if writer is not None:
        writer.close()                                      # zapis konca pliku mp4
        print("Film gotowy:", saved, "klatek =", round(saved / FILM_FPS, 1), "s")
    elif frames_dir is not None:
        print("Klatki png w", frames_dir, "- film: ffmpeg -framerate", FILM_FPS, "-i %05d.png -pix_fmt yuv420p film.mp4")
    if photos_dir is not None:
        print("Zdjecia:", photos, "w", photos_dir)
    writer, frames_dir, photos_dir = None, None, None       # drugie wywolanie nic nie robi

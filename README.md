# Digital Twin – pick-and-place z robotem Franka Panda w NVIDIA Isaac Sim

Projekt pracy dyplomowej: cyfrowy bliźniak stanowiska, na którym robot **Franka Emika Panda** z kamerą głębi **Intel RealSense D435i** na chwytaku opróżnia pudełko z losowo wrzuconymi klockami. Robot sam wykrywa klocki kamerą, wybiera taki, który da się bezpiecznie chwycić, podnosi go palcami i odkłada poza pudełko.

Symulacja jest przygotowaniem do uruchomienia tego samego algorytmu na prawdziwym robocie w laboratorium.

---

## Co potrafi robot

- **Percepcja z kamery głębi:** chmura punktów 3D, wycinanie wnętrza pudełka (ROI), rozdzielanie klocków po skokach wysokości.
- **Położenie, obrót i wymiary klocka:** środek górnej ścianki, kąt obrotu wokół pionu (metoda najmniejszej ramki, błąd poniżej 0.5° względem symulacji) i wymiary z góry (wykrywa sklejone i zasłonięte klocki).
- **Wybór chwytu z kontrolą kolizji:** przed zjazdem sprawdza na chmurze punktów, czy palce i dłoń nie uderzą w ściankę pudełka ani w sąsiedni klocek.
- **Chwyt dopasowany do klocka:** dłoń obraca się zgodnie z kątem klocka; jeśli pełny chwyt jest zablokowany, próbuje płytszego.
- **Ruch po prostej:** zjazd, podniesienie na bezpieczną wysokość i przeniesienie wykonywane jako ścieżka punktów po linii prostej, bez ścinania łuków przez ściankę.
- **Osobne sterowanie ramieniem i palcami.**
- **Pełny cykl** od zdjęcia do odłożenia klocka, powtarzany aż pudełko będzie puste albo nic nie da się chwycić.

---

## Wymagania

- NVIDIA Isaac Sim **6.0.1** (wersja standalone, testowane na Windows)
- karta graficzna NVIDIA RTX
- pakiety Pythona dostępne w środowisku Isaac Sim (`numpy`, `scipy`)

---

## Struktura repozytorium

| Plik | Zawartość |
| --- | --- |
| `scene.py` | budowa sceny (stół, pudełko, klocki, robot, kamera) i wszystkie wymiary fizyczne stanowiska, w tym chwytaka |
| `perception.py` | obraz głębi → chmura punktów → klocki (środek, obrót, wymiary) → wybór chwytu z kontrolą kolizji |
| `kinematics.py` | kinematyka Lula (IK/FK), punkt chwytu (TCP), cele ruchu, ścieżki po prostej |
| `main.py` | pętla symulacji, zrzut klocków do pudełka i maszyna stanów robota |

---

## Uruchomienie

1. Skopiuj pliki do folderu przykładów Isaac Sim, np.
   `standalone_examples\tutorials\getting_started\`
2. Uruchom z folderu instalacji Isaac Sim:

```bat
cd C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64
python.bat standalone_examples\tutorials\getting_started\main.py
```

Po starcie klocki spadają do pudełka, a po ich ułożeniu robot zaczyna pracę. Postęp widać w konsoli (`Chwyt`, `Dojechal`, `Puscil`, `Pominiety` …).

---

## Jak działa algorytm

### Cykl jednego klocka (maszyna stanów)

```
wait → look → move → down → close → up → carry → open → back → look …
                                                              ↘ done
```

| Stan | Co robi |
| --- | --- |
| `wait` | czeka, aż zrzucone klocki się ułożą |
| `look` | gdy ramię stoi w pozycji domowej: zdjęcie, wykrycie klocków, wybór chwytu |
| `move` | jazda nad wybrany klocek z dopasowanym obrotem dłoni |
| `down` | zjazd po prostej do punktu chwytu |
| `close` | zaciśnięcie palców |
| `up` | podniesienie po prostej na bezpieczną wysokość nad ścianką |
| `carry` | przejazd po prostej nad miejsce odkładania, bez obracania nadgarstka |
| `open` | puszczenie klocka |
| `back` | powrót do pozycji domowej |
| `done` | koniec pracy: pudełko puste albo brak klocka do bezpiecznego chwycenia |

### Percepcja

1. Każdy piksel obrazu głębi zamieniany jest na punkt 3D w układzie świata (model kamery otworkowej + pozycja kamery na dłoni).
2. Zostają punkty z wnętrza pudełka (ROI).
3. Piksele na krawędziach (skok wysokości większy niż 1 mm) są odrzucane, a spójne obszary numerowane – każdy obszar to jeden klocek.
4. Dla każdego klocka liczony jest środek górnej ścianki oraz **obrót i wymiary metodą najmniejszej ramki**: prostokąt przymierzany co 1° w zakresie 0–90°, najmniejszy wyznacza kąt klocka.

### Wybór chwytu

Klocki sprawdzane są od najwyższego. Dla każdego program przymierza „szablon” chwytaka nad klockiem i sprawdza w chmurze punktów:

- **pas palców** – czy nic nie wystaje powyżej końca palca,
- **obszar dłoni** – czy nic nie wystaje powyżej spodu dłoni.

Kandydaci: kąt klocka i kąt klocka − 90° (mniejszy obrót najpierw), przy pełnej i płytszej głębokości chwytu. Pierwszy wolny kandydat wygrywa, a klocki bez wolnego chwytu są wypisywane jako `Pominiety` z powodem (`palec` / `dlon`).

### Ruch

- **TCP:** cele podawane są dla środka między palcami; przesunięcie względem `panda_hand` (`TCP_OFFSET = 0.1 m`) zmierzone z ramki `right_gripper` modelu Lula.
- **Ruch po prostej:** `line_path` dzieli odcinek na kroki i dla każdego liczy IK z poprzednim rozwiązaniem jako punktem startowym.
- **Kończenie ruchu:** stan kończy się, gdy przeguby dojadą do celu (`arm_at`), z limitem czasu jako zabezpieczeniem.

---

## Najważniejsze parametry

| Parametr | Plik | Znaczenie |
| --- | --- | --- |
| `ITEM_COUNT`, `ITEM_SIZE` | scene.py | liczba i rozmiar klocków |
| `BOX_*` | scene.py | wymiary i położenie pudełka |
| `FINGER_*`, `HAND_*` | scene.py | wymiary chwytaka zmierzone z modelu (do weryfikacji suwmiarką na robocie) |
| `FINGER_PREOPEN` | scene.py | otwarcie palców przed chwytem |
| `PLACE_POS` | scene.py | miejsce odkładania klocków |
| `GRIP_MARGIN` | scene.py | zapas bezpieczeństwa przy sprawdzaniu kolizji |
| `TCP_OFFSET` | kinematics.py | odległość `panda_hand` → środek między palcami |
| `GRASP_DEPTH`, `SHALLOW_DEPTH` | kinematics.py | pełna i płytka głębokość chwytu |
| `SAFE_Z` | kinematics.py | wysokość przejazdów nad pudełkiem |
| `STEP_LEN`, `CARRY_STEP` | kinematics.py | długość kroku ścieżki (zjazd/podjazd oraz przenoszenie) |
| `MOVE_FRAMES`, `GRIP_FRAMES`, `STEP_FRAMES`, `CAM_DELAY` | main.py | tempo i limity czasowe stanów (w klatkach symulacji) |
| `REMOVE_WALLS` | main.py | test bez ścianek: ścianki przesuwane poza stół po ułożeniu klocków |
| `rng = np.random.default_rng(0)` | main.py | ziarno losowania – ten sam numer daje ten sam układ klocków |

---

## Wyniki (stan obecny)

- Obrót klocka z kamery: **błąd ≤ 0.4°** względem prawdziwego obrotu w symulacji.
- Układ `rng(0)`, 10 klocków z losowym obrotem:
  - **z pudełkiem:** 4–5 z 10 wyjętych,
  - **bez ścianek** (`REMOVE_WALLS = True`): **10 z 10** (11 prób).

Eksperyment kontrolny bez ścianek pokazuje, że głównym ograniczeniem są ścianki pudełka w połączeniu z wielkością dłoni chwytaka, a nie percepcja ani dobór chwytu.

---

## Znane ograniczenia

- Klocek leżący na dnie przy ściance: dłoń (20.8 × 6.3 cm) uderzyłaby w krawędź ścianki – klocek jest pomijany.
- Ciasno ułożone klocki i klocki w narożnikach.
- Sklejone klocki tej samej wysokości są widziane jako jeden obiekt.
- Brak sprawdzenia, czy chwyt się udał (nieudana próba liczy się jak udana).
- Klocki są zrzucane w miejscu odkładania zamiast odkładane.
- Brak planera ruchu dla przeszkód poza pudełkiem.

---

## Plan dalszych prac

- [ ] Przesunięty chwyt dla klocków przy jednej ściance
- [ ] Sprawdzanie chwytu po szerokości palców i ponowna próba
- [ ] Rozsuwanie zbitych i przechylonych klocków palcami
- [ ] Odkładanie klocków zamiast zrzucania
- [ ] Płynny profil prędkości (łagodny start i hamowanie)
- [ ] Unikanie przeszkód
- [ ] Klocki prostopadłościenne
- [ ] Rozdzielenie logiki od warstwy sprzętowej i uruchomienie na prawdziwym robocie Franka w laboratorium

---

## Autor

Maksymilian – praca dyplomowa, symulacja w NVIDIA Isaac Sim.

# Digital Twin — automatyczne wyjmowanie klocków z pojemnika

Praca inżynierska. Symulacja stanowiska zrobotyzowanego w NVIDIA Isaac Sim: ramię
Franka Emika Panda z kamerą głębi zamontowaną na nadgarstku wykrywa klocki
w pojemniku i podjeżdża nad wybrany element.


## Struktura projektu

```
scene.py         budowa sceny: stół, pojemnik, klocki, robot, kamera
perception.py    model kamery otworkowej, chmura punktów, segmentacja klocków
kinematics.py    solver Lula, kinematyka prosta i odwrotna
main.py          pętla symulacji i maszyna stanów
```

### scene.py

Tworzy scenę USD i zwraca uchwyty do robota, klocków i czujnika.

- blat na wysokości `TABLE_SURFACE_Z = 0.75`
- pojemnik: środek `(0.5, 0.0)`, wnętrze 0.4 m, ścianki 0.01 m, wysokość 0.11 m
- 10 klocków o boku 0.05 m, startowo poza polem widzenia
- konfiguracja optyki kamery: ogniskowa wyliczona z zadanego HFOV,
  zakres odcięcia ustawiony na `0.05–10.0 m`

### perception.py

Pełny model kamery otworkowej zaimplementowany od podstaw.

1. `get_intrinsics` — parametry wewnętrzne `(fx, fy, cx, cy)` odczytane z USD
2. `depth_to_pointcloud` — odwrócenie rzutowania: z pary (piksel, głębia)
   do punktu 3D w układzie świata
3. `find_object` — odcięcie obszaru pojemnika, rozdzielenie klocków leżących
   na sobie po skoku głębi, etykietowanie spójnych obszarów
   (`scipy.ndimage.label`), zwrócenie środka każdego obiektu

Obszar zainteresowania (ROI) wyprowadzony geometrycznie z wymiarów pojemnika,
a nie dobrany ręcznie.

### kinematics.py

Warstwa kinematyki oparta na solverze Lula.

- `build_solver` — wczytuje `robot_descriptor.yaml` i `lula_franka_gen.urdf`
  z zasobów Isaaca, ustawia pozę bazy robota na wysokości blatu
- `get_pose` — kinematyka prosta dla zadanych kątów przegubów
- `solve_ik` — kinematyka odwrotna z zadaną pozycją i orientacją,
  wynik rozszerzony z 7 do 9 DOF (palce z `HOME_POSE`)
- `above_target` — punkt podjazdu nad celem

Lula rozwiązuje IK jako zadanie optymalizacji: przybliżenie metodą cyklicznego
zstępowania po współrzędnych (CCD), następnie dopracowanie metodą BFGS,
z wielostartowym próbkowaniem punktów początkowych.

### main.py

Pętla symulacji z maszyną stanów.

## Maszyna stanów

```
wait ──► look ──► move ──► back
          ▲                  │
          └──────────────────┘
```

| Stan | Działanie | Warunek wyjścia |
|---|---|---|
| `wait` | brak | upłynęło `SETTLE_FRAMES` od ostatniego zrzutu |
| `look` | percepcja, wybór najwyższego klocka | znaleziono co najmniej jeden obiekt |
| `move` | IK i rozkaz dla silników (1. klatka), potem oczekiwanie | upłynęło `MOVE_FRAMES` |
| `back` | powrót do `HOME_POSE` | upłynęło `MOVE_FRAMES` |

Percepcja działa wyłącznie w stanie `look`. Ponieważ kamera jest zamontowana
na nadgarstku, pomiar podczas ruchu ramienia byłby bezużyteczny — po podjechaniu
nad klocek kamera widzi wyłącznie jego górną ściankę.

Licznik `state_frame` zerowany jest do `-1`, ponieważ inkrementacja następuje
na początku pętli — dzięki temu pierwsza klatka nowego stanu ma wartość `0`.

## Uruchomienie

```
cd C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64
python.bat <ścieżka>\main.py
```

## Parametry konfiguracyjne

W `main.py`:

| Stała | Wartość | Znaczenie |
|---|---|---|
| `DROP_EVERY` | 15 | odstęp między zrzutami klocków [klatki] |
| `SETTLE_FRAMES` | 30 | czas na opadnięcie klocków [klatki] |
| `MOVE_FRAMES` | 120 | czas na dojazd ramienia [klatki] |
| `TARGET_ALL` | bool | `False` — tylko najwyższy klocek, `True` — wszystkie (tryb testowy) |

W `kinematics.py`:

| Stała | Wartość | Znaczenie |
|---|---|---|
| `APPROACH_HEIGHT` | 0.15 | wysokość podjazdu nad środkiem klocka [m] |
| `LOOK_DOWN` | `euler([0, π, π])` | orientacja chwytaka skierowanego pionowo w dół |

Ziarno generatora losowego ustawione na `0` — układ klocków jest powtarzalny
między uruchomieniami, co pozwala porównywać wersje algorytmu na identycznych
danych wejściowych.

## Weryfikacja

| Sprawdzane | Metoda | Wynik |
|---|---|---|
| Zgodność modelu Luli ze sceną USD | porównanie FK z rzeczywistą pozycją `panda_hand` | różnica < 1 mm |
| Liczba wykrytych klocków | zliczenie po ustabilizowaniu układu | 10/10 |
| Osiągalność celów | IK dla wszystkich wykrytych klocków | 10/10 rozwiązań |
| Powtarzalność obserwacji | trzy kolejne cykle | ta sama pozycja celu (±0.001 m) |

## Stan prac

Zrealizowane:

- [x] Scena, robot, kamera, pojemnik
- [x] Model kamery otworkowej i chmura punktów
- [x] Wykrywanie i rozdzielanie klocków
- [x] Kinematyka prosta, zweryfikowana ze sceną
- [x] Kinematyka odwrotna z zadaną orientacją
- [x] Maszyna stanów, cykl obserwacja–podjazd–powrót

Do zrobienia:

- [ ] Chwytak podciśnieniowy i faza chwytania
- [ ] Zejście do klocka i podniesienie
- [ ] Odkładanie klocków poza pojemnik
- [ ] Obsługa klocków ułożonych nierówno (wyznaczanie normalnej powierzchni)
- [ ] Planowanie trajektorii z omijaniem ścianek pojemnika

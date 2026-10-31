# Digital Twin – pick-and-place z robotem Franka Panda w NVIDIA Isaac Sim

Projekt pracy inżynierskiej: cyfrowy bliźniak stanowiska, na którym robot **Franka Emika Panda** z kamerą głębi **Intel RealSense D435i** na chwytaku zdejmuje sześcienne klocki 5 × 5 × 5 cm leżące na sobie, także pochylone. Robot sam wykrywa klocki kamerą, wybiera taki, który da się bezpiecznie chwycić, i w zależności od trybu odkłada go obok albo układa z klocków wieże.

Symulacja jest przygotowaniem do uruchomienia tego samego algorytmu na prawdziwym robocie w laboratorium.

---

## Co potrafi robot

- **Percepcja z kamery głębi:** chmura punktów 3D, wycinanie obszaru chwytania, podział na ścianki po skokach wysokości i po załamaniu powierzchni.
- **Ścianki pochylone:** normalna ścianki z SVD; krawędzie, obrót i wymiary liczone w płaszczyźnie samej ścianki, więc działa też dla klocków opartych o sąsiada.
- **Sklejone klocki:** ścianka z kilku stykających się klocków w rzędzie dzielona na pojedyncze klocki.
- **Kąt podejścia α:** przy pochylonym klocku dłoń obraca się wokół osi palców tak, żeby schodzić jak najbliżej pionu, a palce trzymają dwie boczne ścianki klocka.
- **Wybór chwytu z kontrolą kolizji i zasięgu:** dla każdego wariantu chwytu sprawdza na chmurze punktów palce i dłoń, a kinematyką odwrotną zasięg ramienia i zjazd bez przekręcania przegubów.
- **Sprawdzanie chwytu** po szerokości palców po zaciśnięciu, po podniesieniu i po przeniesieniu; po 3 pustych chwytach z rzędu robot kończy pracę.
- **Ruch po prostej** przy zjeździe, podnoszeniu i przenoszeniu; przelot nad wieżę z płynnym obrotem dłoni i kontrolą przeskoku przegubu.
- **Dwa tryby odkładania:** `box` – klocki zrzucane w jedno miejsce obok pudełka, `towers` – układanie kilku wież w rzędzie.
- **Wieże:** plan odłożenia z 24 możliwych ustawień sześcianu w dłoni, test wolnej drogi nad wieżą, korekta uchybu przed puszczeniem, zdjęcie wież i kontrola każdego odłożonego klocka.
- **Stanowisko testowe z rampami** o znanym kącie (20°, 30°, 40°) do sprawdzania chwytu pochylonych klocków.

---

## Wymagania

- NVIDIA Isaac Sim **6.0.1** (wersja standalone, testowane na Windows)
- karta graficzna NVIDIA RTX (testowane na RTX 3050, 4 GB)
- pakiety Pythona dostępne w środowisku Isaac Sim (`numpy`, `scipy`)

---

## Struktura repozytorium

| Plik | Zawartość |
| --- | --- |
| `scene.py` | budowa sceny (stół, pudełko, klocki, rampy, robot, kamera) i wszystkie wymiary stanowiska, w tym chwytaka; tryb odkładania i układ wież |
| `perception.py` | obraz głębi → chmura punktów → ścianki (środek, normalna, krawędzie, wymiary) → wybór chwytu z kontrolą kolizji i zasięgu |
| `kinematics.py` | kinematyka Lula (IK/FK), orientacja dłoni z kątem α, punkt chwytu (TCP), ścieżki po prostej i ścieżki z obrotem dłoni |
| `towers.py` | układanie wież: kolejność budowy, stan wież ze zdjęcia, plan odłożenia, test wolnej drogi, kontrola po odłożeniu |
| `main.py` | pętla symulacji, zrzut klocków i maszyna stanów robota |

---

## Uruchomienie

1. Skopiuj pliki do folderu przykładów Isaac Sim, np.
   `standalone_examples\tutorials\getting_started\`
2. Uruchom z folderu instalacji Isaac Sim:

```bat
cd C:\isaacsim\isaac-sim-standalone-6.0.1-windows-x86_64
python.bat standalone_examples\tutorials\getting_started\main.py
```

Po starcie klocki spadają do pudełka, ścianki pudełka są usuwane (`REMOVE_WALLS`), a po ułożeniu się klocków robot zaczyna pracę. Postęp widać w konsoli (`Chwyt`, `Plan odlozenia`, `Kontrola wiezy`, `Pominiety` …), a na końcu podsumowanie: `KONIEC - wyjete: … puste chwyty: …` i stan wież.

Najczęściej zmieniane ustawienia:

| Co | Gdzie |
| --- | --- |
| tryb: zrzut obok albo wieże | `PLACE_MODE = "box"` / `"towers"` w `scene.py` |
| liczba wież i klocków w wieży | `TOWER_COUNT`, `TOWER_HEIGHT` w `scene.py` |
| liczba klocków | `ITEM_COUNT` w `scene.py` |
| inny losowy układ klocków | `SEED` w `main.py` (ten sam numer = ten sam układ) |
| stanowisko z rampami | `TEST_RAMPS = True` w `scene.py` |
| chwyt bez kąta α (do porównania) | `USE_ALPHA = False` w `perception.py` |

---

## Jak działa algorytm

### Maszyna stanów

```
tryb "box":
wait → look → move → down → close → up → carry → open → back → look …

tryb "towers":
wait → tview → back → look → move → down → close → up → carry → place → settle → open → leave → tview → back → look …
```

Pusty chwyt po podniesieniu kończy cykl powrotem do pozycji domowej (`back`). Gdy nie ma już czego chwycić, robot przechodzi do `done`.

| Stan | Co robi |
| --- | --- |
| `wait` | czeka, aż zrzucone klocki się ułożą; usuwa ścianki pudełka i czeka jeszcze raz |
| `look` | gdy ramię stoi w pozycji domowej: zdjęcie, wykrycie ścianek, wybór chwytu, w trybie wież także plan odłożenia |
| `move` | jazda nad wybrany klocek z docelową orientacją dłoni, palce lekko otwarte |
| `down` | zjazd po prostej do punktu chwytu |
| `close` | zaciśnięcie palców i sprawdzenie szerokości chwytu |
| `up` | podniesienie po prostej na wysokość przelotu |
| `carry` | przelot nad miejsce odkładania; w trybie wież z płynnym obrotem dłoni do pozy odłożenia |
| `place` | zjazd po prostej nad górę wieży |
| `settle` | czeka, aż ramię stanie, mierzy uchyb punktu chwytu i w razie potrzeby poprawia pozycję |
| `open` | puszczenie klocka |
| `leave` | odjazd pionowo w górę tą samą drogą |
| `tview` | pozycja oglądania wież: kamera pionowo nad rzędem, zdjęcie i kontrola ostatniego odłożenia |
| `back` | powrót do pozycji domowej |
| `done` | koniec pracy i podsumowanie |

### Percepcja

1. Każdy piksel obrazu głębi zamieniany jest na punkt 3D w układzie świata (model kamery otworkowej + pozycja kamery na dłoni).
2. Zostają punkty z obszaru chwytania (ROI): wnętrze pudełka z zapasem, od 3 cm nad blatem do wysokości 4 warstw klocków.
3. Piksele na krawędziach są odrzucane: skok wysokości większy niż 1 mm albo załamanie powierzchni, czyli normalne sąsiednich pikseli różniące się o więcej niż 10°.
4. Spójne obszary są numerowane (`scipy.ndimage.label`); każdy obszar to jedna ścianka.
5. Dla każdej ścianki liczony jest środek, normalna (SVD) oraz krawędzie i wymiary **metodą najmniejszej ramki** w płaszczyźnie ścianki (prostokąt przymierzany co 1° w zakresie 0–90°).
6. Ścianka o długości równej wielokrotności klocka (± 1 cm) dzielona jest na plastry, po jednym na klocek.

### Wybór chwytu

Ścianki sprawdzane są od najwyższej; w trybie wież najpierw płaskie, a pochylone o ponad 30° dopiero, gdy płaskich nie ma. Dla każdej ścianki program przymierza kolejne warianty chwytu:

- palce wzdłuż jednej albo drugiej krawędzi ścianki,
- głębokość chwytu 2,5 cm, 1,5 cm albo 0,8 cm,
- kąt α: podejście najbliższe pionu, połowa tego kąta, podejście wzdłuż normalnej,
- chwyt na środku albo przesunięty o ±1,3 cm.

Dla każdego wariantu sprawdzane jest w chmurze punktów, czy nic nie wystaje w **pasie palców** powyżej końca palca ani w **obszarze dłoni** powyżej jej spodu, a potem kinematyką odwrotną, czy ramię sięgnie nad klocek i zjedzie bez przekręcania przegubów. Pierwszy wariant, który przejdzie, wygrywa. Klocki bez wolnego chwytu są wypisywane jako `Pominiety` z powodem (`palec`, `dlon`, `ik`, `ik dol`).

### Układanie wież

- **Kolejność budowy:** najpierw wieża najdalej od robota, żeby ramię nie przechodziło nad gotowymi.
- **Klocek w dłoni:** z chwytu wiadomo, jak klocek będzie leżał względem dłoni. Program przymierza 24 ustawienia sześcianu na wieży i dla każdego liczy potrzebną orientację dłoni. Wygrywa ta najbliżej pionu i najmniej różna od orientacji przy chwycie, z karą za palce wzdłuż rzędu i za pochylenie w stronę gotowej wieży.
- **Test wolnej drogi:** palce, dłoń, nadgarstek i klocek zjeżdżają pionowo nad wieżę; żaden punkt sceny nie może stać na ich drodze.
- **Wysokość przelotu:** nad najwyższym punktem sceny, z zapasem na to, co wystaje spod dłoni (klocek i palce).
- **Gdy nie ma pozy odłożenia:** klocek zostaje na później i wybierany jest inny. Gdy miejsce na wieży jest zajęte nawet dla najprostszego chwytu 3 razy z rzędu, wieża jest zamykana.
- **Przed puszczeniem:** pomiar uchybu punktu chwytu nad wieżą (ustalony uchyb regulatora PD w symulacji) i do 2 korekt, gdy przekracza 2 mm. Klocek puszczany 4 mm nad wieżą.
- **Kontrola:** po każdym odłożeniu zdjęcie wież z góry. Klocek przesunięty albo na złej wysokości o ponad 5 mm to `OK z odchylka`. Przesunięcie ponad 15 mm, przechył ponad 3° albo przewrócenie zamykają wieżę.

### Ruch

- **TCP:** cele podawane są dla środka między palcami; przesunięcie względem `panda_hand` (`TCP_OFFSET = 0.1 m`) zmierzone z ramki `right_gripper` modelu Lula.
- **Ruch po prostej:** `line_path` dzieli odcinek na kroki i dla każdego liczy IK z poprzednim rozwiązaniem jako punktem startowym.
- **Przelot z obrotem:** `turn_path` łączy ruch po prostej z obrotem dłoni (interpolacja kwaternionów, krok co 10 cm albo co 10°); ścieżka z przeskokiem przegubu o ponad 0,6 rad jest odrzucana.
- **Kończenie ruchu:** stan kończy się, gdy przeguby dojadą do celu, z limitem czasu jako zabezpieczeniem.
- **Tarcie:** palce mają tarcie 1,0 (domyślnie 0,5), żeby klocek nie wysuwał się przy przenoszeniu.

---

## Najważniejsze parametry

| Parametr | Plik | Znaczenie |
| --- | --- | --- |
| `PLACE_MODE` | scene.py | `"box"` – zrzut obok pudełka, `"towers"` – wieże |
| `TOWER_COUNT`, `TOWER_HEIGHT`, `TOWER_GAP`, `TOWER_ROW_START` | scene.py | liczba wież, klocków w wieży, odstęp i położenie rzędu |
| `ITEM_COUNT`, `ITEM_SIZE` | scene.py | liczba i rozmiar klocków |
| `BOX_*` | scene.py | wymiary i położenie pudełka |
| `FINGER_*`, `HAND_*` | scene.py | wymiary chwytaka zmierzone z modelu |
| `FINGER_PREOPEN` | scene.py | otwarcie palców przed chwytem |
| `PLACE_POS` | scene.py | miejsce zrzutu w trybie `box` |
| `GRIP_MARGIN` | scene.py | zapas bezpieczeństwa przy sprawdzaniu kolizji |
| `TEST_RAMPS`, `RAMP_*` | scene.py | stanowisko testowe z rampami |
| `USE_ALPHA` | perception.py | chwyt z kątem α (`False` = zawsze wzdłuż normalnej) |
| `DEPTH_STEP`, `NORMAL_ANGLE` | perception.py | progi krawędzi: skok wysokości i załamanie powierzchni |
| `SPLIT_TOL` | perception.py | tolerancja podziału sklejonych ścianek |
| `GRASP_SHIFT` | perception.py | przesunięcie chwytu |
| `TILT_LATER` | perception.py | w trybie wież ścianki bardziej pochylone brane na końcu |
| `TCP_OFFSET` | kinematics.py | odległość `panda_hand` → środek między palcami |
| `GRASP_DEPTH`, `SHALLOW_DEPTH`, `TOP_DEPTH` | kinematics.py | głębokości chwytu |
| `SAFE_Z` | kinematics.py | najniższa wysokość przelotu |
| `STEP_LEN`, `CARRY_STEP`, `TURN_STEP` | kinematics.py | kroki ścieżek |
| `VIEW_DIST`, `PLACE_GAP` | towers.py | wysokość kamery nad wieżami, wysokość puszczenia klocka |
| `CHECK_XY`, `CLOSE_XY`, `CLOSE_TILT` | towers.py | progi kontroli wieży po odłożeniu |
| `SEED`, `DROP_TILT` | main.py | układ klocków i ich przechylenie przy zrzucie |
| `REMOVE_WALLS` | main.py | ścianki pudełka usuwane po ułożeniu klocków |
| `HOLD_WIDTH`, `MAX_MISSES` | main.py | próg „klocek w palcach” i limit pustych chwytów z rzędu |
| `CORR_TOL`, `MAX_CORR` | main.py | korekta uchybu nad wieżą |
| `SHOW_TRUTH`, `SHOW_START`, `TEST_TOWER_READ` | main.py | wydruki kontrolne i testy (prawdziwe położenia z symulatora) |

---

## Wyniki

Wszystkie testy: 20 klocków zrzuconych losowo, ścianki pudełka usunięte po ułożeniu, 5 losowych układów (seedy 0–4).

| Test | Wynik |
| --- | --- |
| Tryb `box` | **93/100** klocków wyjętych, 0 pustych chwytów na 93 próby |
| Tryb `towers`, 4 wieże po 5 klocków | **93/100** klocków na wieżach, 0 pustych chwytów, żadna wieża nie zamknięta |
| Rampy 20–40°, chwyt bez kąta α (1 przebieg) | 3/9 |
| Rampy 20–40°, chwyt z kątem α (1 przebieg) | **9/9** |
| Obrót klocka z kamery | błąd ≤ 0,4° względem prawdziwego obrotu |

- W obu trybach wszystkie niewyjęte klocki to jeden układ (seed 1): 7 płaskich klocków tak ciasno obok siebie, że między nimi nie mieści się palec (szczeliny 1–2 cm, palec potrzebuje ok. 3,8 cm).
- W trybie `box` kąt α był potrzebny w 6 chwytach na 93 (klocki oparte pod 25–46°).
- Odkładanie na wieże: uchyb nad wieżą przed korektą średnio 3–4 mm, w chwili puszczenia ok. 1 mm; odłożony klocek przesunięty względem planu średnio o 1,9 mm, najwięcej 5,2 mm.

---

## Znane ograniczenia

- **Ciasno ułożone płaskie klocki:** brak rozsuwania klocków palcami (świadomie pominięte).
- **Ścianki pudełka:** testy idą z usuniętymi ściankami. Przy ściankach dłoń (20,8 × 6,3 cm) nie mieści się przy klocku leżącym na dnie.
- **Stykające się klocki tej samej wysokości:** dzielone tylko wtedy, gdy ścianka ma długość wielokrotności klocka.
- **Mocno pochylone klocki** (ok. 45°): chwyt z dużym kątem α bywa niepewny; w trybie wież takie klocki są brane na końcu.
- **Ruch nad klocek i powrót do pozycji domowej** idą wprost w przestrzeni przegubów, bez sprawdzania kolizji (sprawdzane są odcinki po prostej i droga nad wieżę).
- **Tylko sześciany** o boku `ITEM_SIZE`.
- PhysX nie jest w pełni deterministyczny: ten sam seed może dać nieco inny przebieg.

---

## Plan dalszych prac

- [x] Przesunięty chwyt dla klocków przy ściance
- [x] Sprawdzanie chwytu po szerokości palców
- [x] Chwyt pochylonych klocków (kąt α)
- [x] Odkładanie klocków na wieże
- [ ] Uruchomienie na prawdziwym robocie Franka Panda z kamerą D435i na chwytaku (w toku)
- [ ] Wygląd sceny w stylu hali magazynowej (bez zmian w logice)
- [ ] Rozsuwanie zbitych klocków palcami (odłożone)

---

## Autor

Maksymilian – praca inżynierska, symulacja w NVIDIA Isaac Sim.

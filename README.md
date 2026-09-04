# Lokalizacja obiektow kamera 2,5D w Isaac Sim

Symulacja manipulatora Franka Emika Panda
z kamera RealSense D435i zamontowana na nadgarstku.

## Uruchomienie

    python.bat sciezka\do\main.py

## Struktura

- `main.py` - petla symulacji
- `scene.py` - budowa sceny, konfiguracja kamery
- `perception.py` - algorytm lokalizacji

## Stan prac

Lokalizacja klockow w pudelku dziala.
Dokladnosc: 4,7 mm w osi X, 0,3 mm w osi Y, z odleglosci 45 cm.

Nastepny etap: pudelko i wiele klockow.

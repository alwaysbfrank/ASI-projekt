# ASI-projekt

instrukcja z githuba wykładowcy:
# ASI_2022
 Architektury i metodologie wdrożeń systemów SI. 

Przed uruchomieniem kodów stwórz odpowiednie środowisko `conda`.

1. zainstaluj pakiet `conda`
2. pobierz plik `environment.yml`
3. stwórz środowisko: `$ conda env create -f environment.yml`
4. aktywuj środowisko: `$ conda activate ASI`

# docker
docker build -f Dockerfile .
docker run {container id}

# przebieg
1. przygotowanie danych początkowych (wydzielenie 'batchy', które będą symulować napływanie nowych danych)
2. wytrenowanie modelu (jaki model?)
3. sprawdzenie dryfu na dwóch przewidywanych danych (na dwóch metrykach każda - r2 i rbms(?)) dla nowego batchu
4. jeśli wystapił dryf to wytrenowanie nowego modelu na wszystkich danych, które spłynęły do tej pory

# struktura
struktura uruchomienia -> docker buduje obraz z użyciem miniconda3, która instaluje pythona i inne zależności

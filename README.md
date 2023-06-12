# Licence Plate Recognition
## Opis projektu
Projekt ma na celu identyfikację znaków na blachach rejestracyjnych. W celu osiągnięcia tego celu, wykorzystuje się różne techniki przetwarzania obrazów oraz modele uczenia maszynowego.

## Kroki przetwarzania obrazu
1. Importowanie niezbędnych bibliotek, takich jak numpy, cv2, os, glob oraz tensorflow.
2. Ładowanie modelu uczenia maszynowego.
3. Definiowanie funkcji increase_brightness_contrast, która zwiększa jasność i kontrast obrazu.
4. Definiowanie funkcji perform_processing, która wykonuje przetwarzanie obrazu.
5. Zdefiniowanie banku cech (zakresów jasności i kontrastu) dla przetwarzania obrazu.
6. Określenie parametrów dla filtrów Gaussa oraz rozmiaru docelowego obrazu.
7. Wczytanie obrazu, zmiana jego rozmiaru i konwersja do odcieni szarości.
8. Iteracja po banku cech i stosowanie filtrów Gaussa oraz progowania do wyodrębnienia znaków.
9. Wykrywanie konturów i filtracja znalezionych obszarów na podstawie kryteriów takich jak proporcje i powierzchnia.
10. Transformacja perspektywiczna dla znalezionych obszarów i zamiana ich na obrazy czarno-białe.
11. Wykrywanie pojedynczych liter w znalezionych obszarach i klasyfikacja za pomocą modelu uczenia maszynowego.
12. Złożenie identyfikowanych liter w jeden ciąg i zwrócenie wynikowego tekstu.

## Model uczenia maszynowego rozpoznający znaki
1. Wykorzystano model tf2lite aby skrócić czas ładowania.
2.  Model został wytrenowany na znakach wykorzystywanych przy OCR oraz znakach, z tablic ze zbioru treningowego, które sprawiały największy problem przy rozpoznawaniu.

## Wizualizacja kroków przetwarzania:
1. Zidentyfikowanie tablicy

![image](https://github.com/pawel-gawron/Licence-Plate-Recognition/assets/65308689/64df5c25-8f08-45fb-93f1-31b5f1243531)

2. Transformacja perspektywiczna tablicy

![image_crop](https://github.com/pawel-gawron/Licence-Plate-Recognition/assets/65308689/859585a2-b038-429f-94af-2def8dc823e0)

3. Zidentyfikowanie liter

![image crop letter: 0](https://github.com/pawel-gawron/Licence-Plate-Recognition/assets/65308689/94edd2ce-122d-44a8-80bf-d0ff19a12e12)
![image crop letter: 1](https://github.com/pawel-gawron/Licence-Plate-Recognition/assets/65308689/f4259ae1-eb45-49b7-82f5-a5d5f45870e7)
![image crop letter: 8](https://github.com/pawel-gawron/Licence-Plate-Recognition/assets/65308689/164e677f-a5ff-4c8e-8b98-e267069c570a)
![image crop letter: 9](https://github.com/pawel-gawron/Licence-Plate-Recognition/assets/65308689/ffd29699-4006-4298-9689-8a99ea02a23c)
![image crop letter: 10](https://github.com/pawel-gawron/Licence-Plate-Recognition/assets/65308689/d88ccf67-8b45-40c6-b5d9-ed79c0425f54)
![image crop letter: 11](https://github.com/pawel-gawron/Licence-Plate-Recognition/assets/65308689/d8e4d937-4961-49b2-b495-85724c953dd9)
![image crop letter: 12](https://github.com/pawel-gawron/Licence-Plate-Recognition/assets/65308689/705149a2-37c0-4cea-9a23-662e73c33646)

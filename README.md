# ConeDystrophy Genetic Analyzer 1.0

Gotowa aplikacja desktopowa w Pythonie/PySide6 do badawczej analizy danych genetycznych pacjentów z dystrofią czopków i pokrewnymi chorobami siatkówki.

## Co zawiera
- import CSV, TSV, TXT, XLSX, XLS, VCF i VCF.GZ;
- automatyczne sugerowanie mapowania kolumn oraz ręczna korekta;
- zachowanie warstwy RAW oraz CLEAN;
- standaryzację chromosomów, genów, genotypów, zygotyczności, REF/ALT i pozycji;
- kontrolę jakości: braki, błędne chromosomy, pozycje, genotypy, REF/ALT i duplikaty;
- czyszczenie danych z historią wykonanych operacji;
- filtrowanie wg pacjenta, genu, chromosomu, zygotyczności, typu wariantu, konsekwencji oraz wyszukiwania globalnego;
- analizę pojedynczego pacjenta i genu;
- ranking wariantów i genów;
- statystyki całego zbioru;
- wykresy: geny, chromosomy, typy wariantów, zygotyczność, warianty/pacjent, heatmapa pacjent×gen;
- eksport Excel, CSV, PNG i raport PDF;
- lokalne zapisywanie i otwieranie projektu;
- ustawienie GRCh38/GRCh37 i limitu podglądu.

## Uruchomienie w Windows
Najprościej kliknij `run_windows.bat`. Przy pierwszym uruchomieniu zostanie utworzone środowisko `.venv` i zainstalowane biblioteki.

Ręcznie:
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Test
Wczytaj `sample_data/synthetic_variants.csv`, następnie kliknij `Zastosuj mapowanie i standaryzację`, przejdź do `Kontrola jakości` i uruchom QC.

## Bezpieczeństwo
Program działa lokalnie i samodzielnie nie wysyła danych do internetu. Do analiz zalecane są pseudonimizowane identyfikatory pacjentów. Program jest narzędziem badawczo-analitycznym i nie wykonuje automatycznej diagnozy klinicznej.

## VCF
Wbudowany parser odczytuje podstawowe pola VCF oraz GT/DP/AD. Jeśli INFO zawiera `GENE`, `SYMBOL`, `Gene`, `Consequence`, `ANN` lub `CSQ`, program podejmuje próbę ich zachowania. Złożone adnotacje VEP/ANN można później rozbudować o parser zależny od konkretnego pipeline'u.


## v1.4 — poprawka mapowania kolumn

Zakładka „Import i mapowanie” posiada teraz osobny pionowo przewijany panel mapowania kolumn.
Dzięki temu pola nie nachodzą na siebie przy dużej liczbie kolumn. Dodano też minimalną szerokość pól wyboru i nowoczesny pionowy scrollbar.

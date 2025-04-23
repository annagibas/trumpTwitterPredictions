# Exploratory Data Analysis (EDA) – Raport

## Dane: Trump_Data.csv
- Liczba rekordów: 11 759
- Liczba kolumn: 11

## 1. Struktura danych
Dane zawierają zmienną tekstową `text` oraz 10 zmiennych numerycznych.

## 2. Braki danych
Brak braków danych (wszystkie kolumny mają 100% pokrycia).

## 3. Korelacje
Silna korelacja między `retweet_count` i `favorite_count`.
Zmienna `Followers` również mocno związana z aktywnością tweetów.

## 4. Wartości odstające
Zidentyfikowano outliery w liczbach retweetów i polubień.

## 5. Wnioski
- Dane są kompletne.
- Warto rozważyć przekształcenia logarytmiczne dla dużych wartości.
- Wstępnie: `Hour`, `Day`, `Followers` mogą mieć wartość predykcyjną.
# Feature Engineering – Raport

## Cel  
Celem tego etapu było przygotowanie danych do modelowania poprzez wybór najistotniejszych cech, stworzenie nowych zmiennych oraz transformację zmiennych o nieregularnym rozkładzie (outliery, wartości skrajne).

## Wybór zmiennych oryginalnych

Na podstawie analizy korelacji oraz wykresów boxplot wybrano następujące zmienne:

### Zmienione i przekształcone:
- `retweet_count` → `log_retweet_count`  
- `favorite_count` → `log_favorite_count`  
- `Follower_Change` → `log_follower_change`  
- `Num_Tweets` → `log_num_tweets`

### Zachowane bez zmian:
- `Followers` – bezpośrednia wartość liczby obserwujących  
- `Year` – zmienna trendowa  
- `Hour` – potencjalny wpływ godziny publikacji  

### Dodane nowe cechy binarne:
- `is_weekend` – czy tweet został opublikowany w weekend  
- `is_night` – czy tweet został opublikowany w godzinach nocnych (0–6)  

### Odrzucone zmienne:
- `Month`, `Week`, `Day` – niska korelacja z celem, niewielka wartość informacyjna  
- `text` – nie wykorzystywana w obecnej wersji modelu  
- `Num_Tweets` w wersji surowej – zastąpione wersją logarytmiczną

## Transformacje

### Przekształcenia logarytmiczne (log1p):
Zastosowano funkcję `np.log1p()` (czyli log(x+1)) do zmiennych o dużym rozrzucie i silnej prawoskośności. Ma to na celu:
- zmniejszenie wpływu outlierów,
- poprawę rozkładu,
- ułatwienie działania modeli regresyjnych i drzew decyzyjnych.

#### Uzasadnienie użycia transformacji `np.log1p()`

Transformacje logarytmiczne zostały wykonane przy użyciu funkcji `np.log1p()`, czyli logarytmu naturalnego z wartości `x + 1`. Oznacza to:

Decyzja o użyciu `np.log1p()` zamiast klasycznego `np.log(x)` wynikała z następujących powodów:

### 1. Obsługa wartości równych zero
W zbiorze `Trump_Data.csv` zmienne takie jak `retweet_count`, `favorite_count`, `Follower_Change` i `Num_Tweets` mogą przyjmować wartość `0`. Funkcja `np.log1p(0)` zwraca poprawnie `0`, natomiast `np.log(0)` powodowałaby błąd (log(0) = -∞).

### 2. Większa stabilność numeryczna
Funkcja `np.log1p()` działa dokładniej niż `np.log(x + 1)` przy bardzo małych wartościach zmiennych, ponieważ została zoptymalizowana pod kątem dokładnych obliczeń zmiennoprzecinkowych.

### 3. Możliwość stosowania po przekształceniu wartości ujemnych
Zmienna `Follower_Change` zawiera zarówno wzrosty, jak i spadki liczby obserwujących. Po zastosowaniu wartości bezwzględnej (`abs()`), mogą pojawić się wartości równe 0. `np.log1p()` umożliwia bezpieczne przekształcenie takich danych.

### 4. Uniwersalność i dobre praktyki
`np.log1p()` to powszechnie stosowana funkcja w preprocessing danych przy modelach predykcyjnych. Jej użycie zmniejsza ryzyko błędów i nieoczekiwanych wartości, a także zapewnia spójność i odporność transformacji na zmienne o niskich wartościach.

### Przykład porównania

| Wartość | `np.log(x)` | `np.log1p(x)` |
|--------:|-------------|----------------|
| 0       | błąd        | 0              |
| 1       | 0.0         | 0.693          |
| 10      | 2.302       | 2.398          |
| 1000    | 6.907       | 6.908          |

Wnioski: `np.log1p()` daje wynik bardzo zbliżony do `log(x)`, ale jest bezpieczniejsza i bardziej uniwersalna przy rzeczywistych danych.
Zmienne poddane transformacji:
- `retweet_count`
- `favorite_count`
- `Follower_Change` (wartości brane bezwzględnie, by zachować dodatniość)
- `Num_Tweets`



## Nowe cechy

### is_weekend  
Wartość logiczna, która określa, czy tweet został opublikowany w sobotę lub niedzielę. Obliczono na podstawie kolumny `Week`.

### is_night  
Wartość logiczna, która określa, czy tweet został opublikowany między godziną 0 a 6 rano.

## Finalny zestaw cech

| Nazwa cechy           | Typ                        | Źródło                       |
|------------------------|-----------------------------|-------------------------------|
| `log_retweet_count`    | liczba zmiennoprzecinkowa   | transformacja logarytmiczna |
| `log_favorite_count`   | liczba zmiennoprzecinkowa   | transformacja logarytmiczna |
| `log_follower_change`  | liczba zmiennoprzecinkowa   | transformacja logarytmiczna |
| `log_num_tweets`       | liczba zmiennoprzecinkowa   | transformacja logarytmiczna |
| `Followers`            | liczba całkowita            | zmienna oryginalna          |
| `Year`                 | liczba całkowita            | zmienna oryginalna          |
| `Hour`                 | liczba całkowita            | zmienna oryginalna          |
| `is_weekend`           | zmienna binarna (0/1)       | nowa cecha                  |
| `is_night`             | zmienna binarna (0/1)       | nowa cecha                  |

## Uzasadnienie decyzji dotyczących inżynierii cech

Decyzje podjęte w tym etapie były oparte na wynikach analizy EDA, w szczególności:
- strukturze rozkładów zmiennych (prawoskośność, obecność outlierów),
- korelacjach między zmiennymi,
- logice zjawisk związanych z aktywnością Trumpa na Twitterze.

### Przekształcenia logarytmiczne

Zmienne `retweet_count`, `favorite_count`, `Follower_Change`, `Num_Tweets` zostały przekształcone logarytmicznie za pomocą `np.log1p()` (czyli log(x+1)). Było to uzasadnione:
- silną prawoskośnością tych zmiennych,
- obecnością dużej liczby outlierów,
- znaczną rozpiętością skali (od kilku do kilkuset tysięcy),
- potrzebą ustabilizowania wariancji i zmniejszenia wpływu ekstremów na modele.

Transformacja logarytmiczna poprawia również działanie modeli liniowych i może zwiększyć ogólną skuteczność predykcyjną.

### Zachowane zmienne

`Followers` – zmienna silnie związana z liczbą reakcji (retweetów i polubień). Ma istotne znaczenie merytoryczne i statystyczne.

`Year` – silnie skorelowana z rosnącym zasięgiem tweetów. Umożliwia modelowi uchwycenie zmian w czasie.

`Hour` – mimo braku silnych korelacji, została zachowana ze względu na możliwy wpływ pory publikacji na skuteczność tweetów.

### Nowe zmienne logiczne

`is_weekend` – informuje, czy tweet został opublikowany w weekend (sobota lub niedziela). Użytkownicy mogą inaczej reagować w dni wolne.

`is_night` – wskazuje, czy tweet pojawił się między godziną 0 a 6. Tweetowanie nocne może przyciągać mniej lub bardziej uwagi.

Obie zmienne są binarne i gotowe do bezpośredniego użycia w modelach.

### Zmienne odrzucone

`Month`, `Week`, `Day` – wykazały bardzo niską korelację z celem i nie wnosiły istotnej wartości informacyjnej. Ich pozostawienie mogłoby wprowadzać szum do modelu.

`text` – nie została uwzględniona w bieżącym etapie, ponieważ wymagałaby osobnego podejścia (NLP, embeddingi). Może być wykorzystana w kolejnych etapach rozwoju projektu.
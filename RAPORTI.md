# RAPORTI — UrbanBike AI

## Autori

| Fusha | Vlera |
| --- | --- |
| Emri Mbiemri | Erdiona Tola |
| Grupi | Grupi 22 |
| Roli në aktivitet | Analiza e te dhenave dhe modelimi |
| Figma | _(do të shtohet)_ |
| GitHub-i im | https://github.com/tolaerdiona-commits/urbanbike-ai-erdiona-tola |

## Qëllimi

**Problemi që zgjidhim:** Sa qira biçikletash priten për ditën pasuese në Washington, D.C.?

**Target-i:** `rentals` (totali ditor i qirave, kolona `cnt` te UCI).

Të dhënat janë historike (2011–2012): UCI Bike Sharing (`data/raw/day.csv`) dhe moti ditor
NASA POWER (`data/raw/nasa_weather.csv`: `temp_c` °C, `rain_mm` mm/ditë, `wind_ms` m/s),
të bashkuara sipas `date`.

## Pastrimi

| Kontrolli | Vlera |
| --- | --- |
| Origjinali UCI | 731 rreshta, 0 mungesa |
| Rreshta NASA / pas bashkimit | 731 / 731 |
| Kopja didaktike (para pastrimit) | 743 rreshta |
| Dublime identike | 12 |
| Qeliza bosh para pastrimit | 30 |
| Pas pastrimit | 731 rreshta |
| Mungesa meteorologjike që mbeten | 36 |
| Pas lag-eve (dataset-i i modelit) | 724 rreshta |

Problemet u injektuan qëllimisht **vetëm** në kopjen mësimore `data/teaching/dirty.csv`
(seed 42): 18 temperatura bosh, 12 reshje bosh,
6 tekste „gabim” te temperatura dhe 12 rreshta të dublikuar.
Burimet reale te `data/raw/` mbeten të paprekura.

Tri vendimet e pastrimit:

1. **Dublimet:** u hoqën 12 rreshta identikë me `drop_duplicates()`,
   sepse e njëjta ditë nuk duhet të numërohet dy herë. Mbeten 731 ditë unike,
   të renditura në rend kohor dhe pa ditë që mungojnë.
2. **Tipet dhe tekstet e gabuara:** `date` u kthye nga `object` në `datetime64[ns]`, ndërsa
   `temp_c` nga `object` në `float64` me `pd.to_numeric(errors="coerce")`. Tekstet „gabim” u bënë
   NaN, prandaj mungesat meteorologjike janë 36
   (30 boshllëqe + 6 tekste).
   Mungesa **nuk** trajtohet si 0 °C, sepse kjo do të ishte një temperaturë e rreme.
3. **Imputimi vetëm nga train:** mungesat plotësohen me medianën brenda `Pipeline`
   (`SimpleImputer(strategy="median", add_indicator=True)`), e cila mësohet vetëm nga train.
   Nëse mediana llogaritej nga i gjithë dataset-i, informacioni i validation/test do të
   rridhte në trajnim (data leakage).

Lag-et (`rentals_lag1`, `rentals_lag7`, `rentals_mean7`, moti `_lag7`) krijojnë boshllëqe në 7 ditët e para,
prandaj ato hiqen dhe mbeten 724 rreshta.

## Grafikët

Të gjitha figurat përdorin vetëm periudhën train (2011-01-08 – 2012-04-30).

1. **`reports/figures/01_demand.png`** — *Vëzhgim:* qiratë mesatare rriten nga rreth
   1774 në ditë në 90 ditët e para të train në rreth 4914 në 90 ditët e fundit,
   me luhatje të forta nga dita në ditë; mesatarja e kaluar 7-ditore e zbut këtë luhatje.
   *Shpjegim i mundshëm:* sezonaliteti (dimër → pranverë/verë) dhe rritja e përdorimit të shërbimit.
2. **`reports/figures/02_workingday.png`** — *Vëzhgim:* mesatarja në ditë pune është
   3761 qira, ndërsa në fundjavë/festë 3570 qira.
   *Shpjegim i mundshëm:* udhëtimet për punë; mesatarja nuk e përshkruan gjithë shpërndarjen.
3. **`reports/figures/03_weather.png`** — *Vëzhgim:* temperatura NASA e 7 ditëve më parë dhe qiratë
   kanë korrelacion r = 0.54 në train. *Shpjegim i mundshëm:* moti më i ngrohtë lidhet me
   më shumë qira, por lidhja vizuale nuk provon shkakësi.

## Vlerësimi

Ndarja kohore (e ardhmja nuk përzihet me të kaluarën):

| Pjesa | Datat | Rreshta |
| --- | --- | --- |
| Train | 2011-01-08 – 2012-04-30 | 479 |
| Validation | 2012-05-01 – 2012-08-31 | 123 |
| Test | 2012-09-01 – 2012-12-31 | 122 |

**Zgjedhja sipas validation (MAE, më i ulët = më mirë):**

| Modeli | Validation MAE |
| --- | --- |
| Yesterday | 775.203 |
| DummyMean | 2953.866 |
| Ridge | 687.558 |
| RandomForest | 1821.528 |

**Testi final (qira/ditë):**

| Modeli | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Yesterday | 916.402 | 1330.386 | 0.539 |
| DummyMean | 2494.628 | 2790.064 | -1.027 |
| Ridge **(i zgjedhur)** | 893.251 | 1252.655 | 0.591 |
| RandomForest | 1671.499 | 2042.654 | -0.087 |

**Modeli i zgjedhur:** **Ridge**, sepse ka MAE më të ulët në validation
(687.558) ndër dy kandidatët ML. Zgjedhja u bë para leximit të testit dhe
parametrat nuk u ndryshuan sipas testit. Në test Ridge ka MAE 893.251,
rreth 2.5% më pak se baseline-i Yesterday
(916.402). Nuk pretendohet domethënie statistikore.

**Çfarë ndodhi me RandomForest:** MAE në validation 1821.528 dhe në test
1671.499 (R² = -0.087). Random Forest del më keq: pemët nuk mund të ekstrapolojnë përtej vlerave të parë në train, ndërsa kërkesa në 2012 është më e lartë se në train. Një model më kompleks nuk është automatikisht më i mirë.

Figura e testit: `reports/figures/04_predictions.png`.

## Klasifikimi (fakultativ)

Pragu = mediana e qirave në train = **3907** qira. Klasa „E lartë” = rentals ≥ pragut (rregull mësimor).

| Modeli | Accuracy | Balanced accuracy | Macro-F1 | Matrica |
| --- | --- | --- | --- | --- |
| DummyMostFrequent | 84.43% | 50.00% | 0.458 | `[[0, 19], [0, 103]]` |
| LogisticRegression | 91.80% | 77.98% | 0.820 | `[[11, 8], [2, 101]]` |

Testi ka 103 ditë të larta dhe 19 normale, pra klasat janë të pabalancuara. Baseline-i DummyMostFrequent merr accuracy të lartë vetëm duke thënë gjithmonë „E lartë”, por balanced accuracy 50% tregon se nuk dallon asgjë. Prandaj accuracy lexohet bashkë me balanced accuracy dhe macro-F1. Te LogisticRegression 8 ditë normale dalin të larta dhe 2 të larta dalin normale (figura `reports/figures/05_confusion.png`).

## Aplikacioni

`app.py` nis një server lokal në http://127.0.0.1:8000 që përdor modelin real
(`models/regression.joblib`) për datat e testit. Shembull për 2012-09-01:
shih `/api/predict?date=2012-09-01`.

## Kufizimet

1. **Një qytet dhe një periudhë historike:** vetëm Washington, D.C., 2011–2012; rezultatet nuk
   vlejnë automatikisht për qytete të tjera ose për kërkesën e sotme.
2. **Moti NASA POWER:** vjen nga një rrjet meteorologjik i reanalizës me rezolucion të ulët dhe me
   ditë sipas kohës diellore lokale (LST), jo nga një stacion në qytet. Lag7 nuk e zgjidh çdo
   kufizim të reanalizës historike.
3. **Parashikim njëditor historik:** supozohet që numri i qirave të djeshme është i njohur;
   aplikacioni nuk është shërbim i testuar për parashikim të drejtpërdrejtë për sot.

## Kontributet individuale

| Studenti | Skedari | Veprimi | Vendimi im |
| --- | --- | --- | --- |
| Erdiona Tola | 01_clean.py – app.py, RAPORTI.md | Ndërtova dhe ekzekutova të gjithë pipeline-in, interpretova grafikët dhe metrikat. | Zgjodha modelin vetëm sipas MAE në validation dhe dokumentova pse Random Forest nuk e përmirësoi rezultatin. |

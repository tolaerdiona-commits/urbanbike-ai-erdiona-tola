# Raporti i studentit: UrbanBike AI

Studenti: Erdiona Tola
Grupi (nëse punohet në grup): Grupi 22 — roli: Analiza e te dhenave dhe modelimi
Data: 2026-11-12
GitHub-i im: https://github.com/tolaerdiona-commits/urbanbike-ai-erdiona-tola
Figma: _(do të shtohet)_

## Qëllimi

Problemi që zgjidhim: Sa qira biçikletash priten për ditën pasuese në Washington, D.C.?

Target-i y: `rentals` — totali ditor i qirave (kolona `cnt` te UCI).

Predictorët X: `rentals_lag1`, `rentals_lag7`, `rentals_mean7`, `temp_c_lag7`, `rain_mm_lag7`,
`wind_ms_lag7`, `trend_days` (numerikë) dhe `weekday`, `month`, `workingday`, `holiday` (kategorikë).

Koha kur parashikojmë dhe çfarë informacioni është i njohur: parashikojmë një ditë përpara
(parashikim njëditor historik). Dihen qiratë e djeshme dhe të 7 ditëve të kaluara, kalendari i ditës
(dita e javës, muaji, ditë pune/festë) dhe moti NASA i 7 ditëve më parë (`_lag7`). Moti i së njëjtës
ditë **nuk** përdoret, sepse në momentin e parashikimit nuk dihet.

## Pastrimi

Burimet reale dhe çelësi i bashkimit: UCI Bike Sharing (`data/raw/day.csv`, 731 rreshta, 0 mungesa)
dhe moti ditor NASA POWER (`data/raw/nasa_weather.csv`, 731 rreshta: `temp_c` °C, `rain_mm` mm/ditë,
`wind_ms` m/s). U bashkuan sipas kolonës `date` → 731 rreshta.

Dallimi mes origjinaleve dhe kopjes didaktike: burimet te `data/raw/` mbeten të paprekura. Problemet
u injektuan qëllimisht **vetëm** në kopjen mësimore `data/teaching/dirty.csv` (seed 42):
18 temperatura bosh, 12 reshje bosh, 6 tekste „gabim” te temperatura dhe 12 rreshta të dublikuar.

Rreshtat para/pas: 743 rreshta në kopjen didaktike → 731 pas pastrimit → 724 në dataset-in e modelit
(lag-et krijojnë boshllëqe në 7 ditët e para, prandaj ato hiqen).

Dublimet: 12 rreshta identikë u hoqën me `drop_duplicates()`, sepse e njëjta ditë nuk duhet të
numërohet dy herë. Mbeten 731 ditë unike, në rend kohor, pa ditë që mungojnë.

Mungesat para/pas konvertimit: para konvertimit 30 qeliza bosh. `date` u kthye nga `object` në
`datetime64[ns]` dhe `temp_c` nga `object` në `float64` me `pd.to_numeric(errors="coerce")`;
6 tekstet „gabim” u bënë NaN, prandaj pas konvertimit mbeten **36** mungesa meteorologjike
(30 + 6). Mungesa **nuk** trajtohet si 0 °C, sepse kjo do të ishte temperaturë e rreme.

Arsyeja e imputimit vetëm me train: mungesat plotësohen me medianën brenda `Pipeline`
(`SimpleImputer(strategy="median", add_indicator=True)`), që mësohet vetëm nga train. Nëse mediana
llogaritej nga i gjithë dataset-i, informacioni i validation/test do të rridhte në trajnim (data leakage).

## Grafikët

Të gjitha figurat përdorin vetëm periudhën train (2011-01-08 – 2012-04-30).

Vëzhgimi 1 dhe figura përkatëse: `reports/figures/01_demand.png` — qiratë mesatare rriten nga rreth
1774 në ditë në 90 ditët e para të train në rreth 4914 në 90 ditët e fundit, me luhatje të forta nga
dita në ditë; mesatarja e kaluar 7-ditore e zbut këtë luhatje. Shpjegim i mundshëm: sezonaliteti
(dimër → pranverë/verë) dhe rritja e përdorimit të shërbimit.

Vëzhgimi 2 dhe figura përkatëse: `reports/figures/03_weather.png` — temperatura NASA e 7 ditëve më
parë dhe qiratë kanë korrelacion r = 0.54 në train: ditët më të ngrohta lidhen me më shumë qira.
(Plotësues: `reports/figures/02_workingday.png` — mesatarja në ditë pune është 3761 qira, në
fundjavë/festë 3570; mesatarja nuk e përshkruan gjithë shpërndarjen.)

Pse një lidhje e dukshme nuk provon shkakësi: temperatura dhe qiratë rriten të dyja me stinën dhe me
kalimin e kohës (trend). Një faktor i tretë (sezoni, dita më e gjatë, rritja e shërbimit) mund t'i
lëvizë të dyja bashkë, prandaj grafiku tregon vetëm shoqërim, jo që temperatura i shkakton qiratë.

## Vlerësimi

Ndarja kohore (e ardhmja nuk përzihet me të kaluarën): Train 2011-01-08 – 2012-04-30 (479 rreshta),
Validation 2012-05-01 – 2012-08-31 (123), Test 2012-09-01 – 2012-12-31 (122).

| Modeli | Validation MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Yesterday | 775.203 | 916.402 | 1330.386 | 0.539 |
| DummyMean | 2953.866 | 2494.628 | 2790.064 | -1.027 |
| Ridge **(i zgjedhur)** | 687.558 | 893.251 | 1252.655 | 0.591 |
| RandomForest | 1821.528 | 1671.499 | 2042.654 | -0.087 |

Modeli i zgjedhur nga validation dhe arsyeja: **Ridge**, sepse ka MAE më të ulët në validation
(687.558) ndër dy kandidatët ML. Zgjedhja u bë para leximit të testit dhe parametrat nuk u ndryshuan
sipas testit.

Rezultati i provës me Random Forest: MAE në validation 1821.528 dhe në test 1671.499 (R² = -0.087),
më keq edhe se baseline-i Yesterday. Pemët nuk mund të ekstrapolojnë përtej vlerave të parë në train,
ndërsa kërkesa në 2012 është më e lartë se në train. Një model më kompleks nuk është automatikisht më i mirë.

Interpretimi i gabimit në qira për ditë: në test Ridge gabon mesatarisht rreth **893 qira në ditë**
(MAE 893.251), rreth 2.5% më pak se baseline-i Yesterday (916.402). Shembull nga aplikacioni: për
2012-09-01 parashikimi është 6834.73 ndërsa vlera reale 6140 (gabim 694.73). Nuk pretendohet
domethënie statistikore. Figura e testit: `reports/figures/04_predictions.png`.

Rezultati i klasifikimit, nëse u realizua: po. Pragu = mediana e qirave në train = **3907**; klasa
„E lartë” = rentals ≥ pragut.

| Modeli | Accuracy | Balanced accuracy | Macro-F1 | Matrica |
|---|---:|---:|---:|---|
| DummyMostFrequent | 84.43% | 50.00% | 0.458 | `[[0, 19], [0, 103]]` |
| LogisticRegression | 91.80% | 77.98% | 0.820 | `[[11, 8], [2, 101]]` |

Testi ka 103 ditë të larta dhe 19 normale (klasa të pabalancuara). DummyMostFrequent merr accuracy të
lartë vetëm duke thënë gjithmonë „E lartë”, por balanced accuracy 50% tregon se nuk dallon asgjë.
Te LogisticRegression 8 ditë normale dalin të larta dhe 2 të larta dalin normale
(`reports/figures/05_confusion.png`).

## Kufizimet

1. Një qytet dhe një periudhë historike: vetëm Washington, D.C., 2011–2012; rezultatet nuk vlejnë
   automatikisht për qytete të tjera ose për kërkesën e sotme.
2. Moti NASA POWER vjen nga një rrjet reanalize me rezolucion të ulët dhe me ditë sipas kohës diellore
   lokale (LST), jo nga një stacion në qytet; lag7 nuk e zgjidh çdo kufizim të reanalizës historike.
3. Parashikim njëditor historik: supozohet që numri i qirave të djeshme është i njohur; aplikacioni
   `app.py` (http://127.0.0.1:8000) nuk është shërbim i testuar për parashikim të drejtpërdrejtë për sot.

## Kontributet individuale

| Studenti | Skedari/ekrani | Puna dhe vendimi | Commit/prova |
|---|---|---|---|
| Erdiona Tola | 01_clean.py – app.py, RAPORTI.md | Ndërtova dhe ekzekutova të gjithë pipeline-in, interpretova grafikët dhe metrikat. Vendimi: zgjodha modelin vetëm sipas MAE në validation dhe dokumentova pse Random Forest nuk e përmirësoi rezultatin. | Commit-et `94b3913`, `bcc5d42`; prova e app.py: 2012-09-01 → 6834.73 (reale 6140) |

## Burimet

UCI: https://doi.org/10.24432/C5W894
NASA: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
Provenienca lokale: data/raw/provenance.json

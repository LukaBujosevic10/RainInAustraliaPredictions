# Predikcija kiše u Australiji

Ovaj projekat je rađen u okviru predmeta Uvod u nauku o podacima, a cilj nam je bio da napravimo model koji ume da proceni da li će sledećeg dana padati kiša na određenoj lokaciji u Australiji, na osnovu jučerašnjih/današnjih meteoroloških merenja. Radili smo na skupu podataka Rain in Australia, koji broji preko 145.000 dnevnih merenja sa 49 stanica u periodu 2007–2017.

U pitanju je  problem nadgledanog učenja sa binarnom ciljnom promenljivom (`RainTomorrow`), a sve finalne metrike (Accuracy, ROC-AUC, F1, preciznost, odziv) računali smo isključivo na test skupu.

Svi detalji projekta, tj. kompletna analiza, sve ćelije, obrazloženja odluka i grafici se nalaze u fajlu `FinalDocument.ipynb`.

## Pokretanje projekta

Da bi se `FinalDocument.ipynb` mogao izvršiti od početka do kraja, potrebno je sledeće:

- **Python okruženje** - sve biblioteke koje sveska koristi (pandas, numpy, scikit-learn, xgboost, mlflow, networkx, openpyxl, seaborn/matplotlib...) navedene su u `req/requirements.txt`; prva ćelija sveske (`pip install -r req/requirements.txt`) ih instalira automatski. Za tačne verzije koje smo koristili pogledati taj fajl.
- **Ulazni podaci** - potrebno je da u `InputData/` postoje `weatherAUS.csv` (Rain in Australia sa Kaggle-a) i `stations.txt` (spisak meteoroloških stanica sa bom.gov.au/climate/data/lists_by_element/stations.txt), a u `GeoPodaci/` fajl sa razdaljinama između stanica koji se koristi za prostorne atribute i imputaciju.
- **Redosled izvršavanja** - ćelije treba pokretati redom odozgo nadole; svaka faza čuva svoj rezultat u `backups/` kao csv, a naredna faza ga učitava, tako da se sveska ne sme pokretati isečke van redosleda niti paralelno.

## Organizacija projekta

Jupyter notebooks su podeljene po fazama:

1. **Sistemske i fizičke greške** - detekcija i korekcija merenja koja krše fizičke zakone (npr. `MinTemp > MaxTemp`), pre bilo kakve dalje obrade. Ovo ne uči iz podataka, pa ide pre podele.
2. **Eksplorativna analiza podataka (EDA)** - deskriptivna statistika, distribucije, korelacije. Takođe pre podele, ali čisto deskriptivno - ne piše ništa što kasnija faza koristi.
3. **Train/test split** - hronološka podela, granica od koje počinje "no leakage" zona.
4. **Detekcija anomalija** - fit isključivo na trening skupu.
5. **Feature engineering** - ciklično kodiranje, geografske/klimatske karakteristike stanica, lag i pokretni proseci (deterministički računi, ne uče iz podataka - bezbedno i pre i posle splita).
6. **Analiza značaja atributa** - Random Forest feature importance i mutual information, fit isključivo na trening skupu, korišćeno za procenu odnosa nedostajućih vrednosti i prediktivne koristi svakog atributa.
7. **Obrada nedostajućih vrednosti** - fit na trening skupu, transformacija oba skupa.
8. **Skaliranje i treniranje modela** - `StandardScaler` unutar istog sklearn pipeline-a kao model, fit samo na trening skupu.
9. **Produkcija modela** - MLflow serving.

Plus Prilog u kome smo objasnili odbačene eksperimente(i razlog zašto smo ih odbacili), van produkcionog toka.

*Napomena: brojevi faza gore (1-9) su tematski, ne odgovaraju direktno brojevima poglavlja u `FinalDocument.ipynb` (koja idu do 13, plus podpoglavlja) - svesku je bilo prirodnije podeliti na više sitnijih koraka (npr. konfiguracija servisa, učitavanje podataka, analiza značaja atributa su tamo zasebna poglavlja), dok ih ovde grupišemo po logičkoj fazi pipeline-a.*

## Rad sa podacima

Na startu smo imali dosta problema u meteorološkim podacima kao što su: fizički nemoguća merenja (temperature van dnevnog opsega, pritisak koji skače brže nego što fizika dozvoljava, tačka rose viša od trenutne temperature), kolone koje nedostaju kod skoro polovine opažanja, i stanice razbacane po celom kontinentu koje treba nekako geografski povezati.

Ukratko, evo šta smo uradili povodom svakog od ovih problema:

- proverili deset fizičkih pravila (R1-R10) - od očiglednih (`MinTemp` ne sme biti veći od `MaxTemp`, izmerene temperature u 9h/15h moraju biti unutar dnevnog raspona) do suptilnijih (promena pritiska preko 2 hPa/sat je fizički nemoguća, tačka rose mora biti niža od trenutne temperature, `RainToday`/`Rainfall` ne smeju biti u koliziji), uz zvanične rekordne vrednosti sa bom.gov.au kao referencu za ekstremne outlier-e,
- podelili podatke hronološki (granica 10.11.2015.), ne nasumično - vremenska serija se ne sme mešati unapred/unazad,
- detektovali anomalije šestostrukim pristupom (MAD Z-skor, IQR, Mahalanobisova udaljenost, Isolation Forest, LOF, KNN) kombinovanim u jedan težinski skor, sa pragom na 99.9-om percentilu računatim samo na trening delu,
- popunili nedostajuće vrednosti kombinacijom prostorne imputacije (grupe susednih stanica, birane po korelaciji) i Random Forest/IDW modela i to smo uradili tek nakon feature engineering-a, jer neki izvedeni atributi (razlike, pokretni proseci) služe kao prediktori unutar samih imputacionih modela.

Pored ovoga, tokom rada smo primetili i nekoliko zanimljivih obrazaca u podacima, koje ovde ukratko dokumentujemo:

- `Sunshine`, `Evaporation` i `Cloud9am`/`Cloud3pm` nedostaju kod 38-48% opažanja (redom: 48%, 43%, 38%, 41%) - a ipak je Random Forest imputacija postigla R² do 0.98 (najteža kolona, `Sunshine`, R² ≈ 0.79). Vizuelno poređenje pokazuje da medijana pravi veštački "šiljak" tačno na medijani kod stanica koje su nekad imale 100% nedostajućih vrednosti (npr. Albury, BadgerysCreek), dok prostorna/RF imputacija daje uverljivu, po lokaciji specifičnu raspodelu.
- Korelacija stanica po `RainToday` opada predvidljivo sa rastojanjem - prosečno 0.73 na <50km, 0.59 na 100-150km, 0.40 na 200-300km, ~0.01 preko 1000km. Na osnovu ovoga smo probali i nova "susedna" obeležja (ponderisani prosek RainToday%/Pressure3pm okolnih stanica u krugu od 300km), ali su ona u finalnom modelu **izostavljena** - `feature_importances_` je pokazao da doprinose svega 2-3% odluke svako, pa smo ih, radi jednostavnijeg serviranja modela (bez potrebe da se za svaku predikciju uživo prikupljaju podaci sa svih susednih stanica), izbacili iz finalnog skupa atributa. Analiza je ostavljena u svesci kao dokumentacija zašto su uopšte razmatrana.
- Automatsko povezivanje imena stanica iz skupa sa zvaničnim BOM meteorološkim šiframa nije bilo trivijalno - neka imena se razlikuju samo u razmaku/velikim slovima (`Nhil` → `NHILL`), pa je trebalo ručno mapirati nekoliko izuzetaka i hardkodovati tačne BOM ID-jeve za veće gradove da izbegnemo pogrešno uparivanje.
- Redosled faza 5 (feature engineering) i 7 (imputacija) je namerno obrnut u odnosu na "prirodan" redosled - jer se izvedeni atributi iz faze 5 koriste kao prediktori unutar imputacionih modela u fazi 7. Ovo nije curenje podataka jer su sve transformacije u fazi 5 deterministički računi (ciklično kodiranje, razlike, lag/rolling po lokaciji), ne uče ništa iz podataka.

Ceo proces (imputacija, skaliranje, treniranje) je za finalni model upakovan u pipeline koji se fituje samo na trening skupu.

## Modeli i rezultati

Konačan skup za modelovanje (posle uklanjanja redova bez `RainTomorrow` vrednosti): 133.283 opservacije (107.152 train / 26.131 test), uz izraženu neuravnoteženost klasa (~77% dana bez kiše). Zato smo modele poredili uz balansiranje klasa i optimizaciju praga odlučivanja radi maksimizacije F1 ocene za manjinsku (kišnu) klasu.

| Model | Accuracy | ROC-AUC | F1 (kiša) | Preciznost | Odziv |
|---|---|---|---|---|---|
| Decision Tree (balansirano, prag=0.62) | 81.0% | 0.846 | 0.62 | 0.56 | 0.69 |
| Random Forest (balansirano, prag=0.42) | 82.3% | 0.876 | **0.65** | 0.59 | 0.73 |
| **XGBoost** (balansirano, prag=0.34) | 81.1% | **0.878** | 0.64 | 0.56 | **0.76** |
| XGBoost (podrazumevani prag 0.50, bez balansiranja) | 85.0% | 0.879 | 0.62 | 0.73 | 0.54 |

Decision Tree je, očekivano, najslabiji. Random Forest sa balansiranim klasama i optimizovanim pragom ima nešto viši F1, dok XGBoost (balansirano) ima nešto viši ROC-AUC uz malo niži F1 - jeftinije propušta kišu (odziv 0.76 naspram 0.73) ali uz nešto više lažnih uzbuna (preciznost 0.56 naspram 0.59). Za poređenje, XGBoost sa podrazumevanim pragom (bez balansiranja) postiže najvišu ukupnu tačnost i najvišu preciznost, ali znatno lošiji odziv za kišne dane (54%) - očekivan kompromis preciznost/odziv kod neuravnoteženih klasa.

Iako Random Forest ima neznatno viši F1, u produkciju je registrovan **XGBoost (balansirano, prag=0.34)** - ima najviši ROC-AUC, viši odziv za kišnu (manjinsku) klasu, koji nam je bitniji od preciznosti s obzirom da je propuštanje kiše skuplja greška od lažne uzbune, i jednostavniji/konzistentniji skup atributa za serviranje uživo.

Praćenje eksperimenata i produkciono servisiranje najboljeg modela realizovano je preko **MLflow**.

## Ograničenja

Model dosta bolje hvata opšti obrazac (ROC-AUC ≈ 0.88) nego same retke kišne dane. F1 za kišnu klasu ostaje oko 0.65 čak i posle balansiranja i optimizacije praga, a preciznost/odziv kompromis je neizbežan kod ovakve neravnoteže (~77% dana bez kiše). Skup pokriva samo period 2007-2017 i samo australijske stanice, tako da model ne vidi novije klimatske trendove niti lokacije van ovog konteksta. Nedostajuće vrednosti u `Sunshine`/`Evaporation`/oblačnosti su i dalje procena, ne stvarno merenje, posebno kod stanica koje su istorijski imale 100% nedostajućih podataka za neku kolonu.

Kao logičan sledeći korak, ima smisla probati sa dodatnim spoljnim podacima (satelitski snimci oblačnosti, sinoptičke karte), ili modelovati odvojeno po klimatskim zonama s obzirom da Australija pokriva i tropske i umerene i pustinjske uslove.

## Airflow

Pored same analize i treniranja modela, napravili smo i produkcionu verziju celog pipeline-a koristeći Apache Airflow, kao demonstraciju automatizacije data science procesa. Taj deo je odvojen u sopstveni projekat:

**[`../weatheraus-airflow`](https://github.com/LukaBujosevic10/weatheraus-airflow)**

Tamo se nalazi kompletno uputstvo za pokretanje (Docker/Docker Compose), definisan DAG koji redom pokreće produkcioni podskup svezaka, MLflow za praćenje eksperimenata i model registry, FastAPI servis za serviranje predikcija (hvata stvarne podatke današnjeg dana sa australijskih stanica) i Streamlit aplikacija sa mapom stanica. Sam Airflow, njegovo korišćenje i specifičnosti su detaljno opisane u njegovom README.md.
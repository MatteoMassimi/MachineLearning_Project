# Task 1 — Preparazione del dataset (Bank Marketing)

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** Bank Marketing (UCI) — `bank-additional-full.csv`
**Codice di riferimento:** `src/task1.ipynb`


---

## 1. Obiettivo del task

Trasformare il dataset grezzo in un formato comprensibile dai classificatori ed
estrarre due file di lavoro, usati nei task successivi:

| File | Dimensione | Scopo |
|------|-----------|-------|
| `manuale.csv` | 12 istanze | calcoli **a mano** dei classificatori (Task 2) |
| `training.csv` | dataset completo pulito | EDA e addestramento (Task 3, 4, 5) |



---

## 2. Riproducibilità

Si fissa il seme dei generatori casuali (`SEED = 10`, `np.random.seed(SEED)`).
Questo garantisce che i file estratti — in particolare il
campionamento di `manuale.csv` — siano **sempre gli stessi** a ogni esecuzione e che
i risultati siano replicabili.

---

## 3. Caricamento

Il file usa il **punto e virgola** (`;`) come separatore, senza specificarlo pandas leggerebbe tutto in un'unica colonna.
Il dataset originale conta 41.188 istanze × 21 colonne (20 attributi + classe `y`).

---

## 4. Controllo qualità di base

### 4.1 Duplicati

Le 12 istanze identiche presenti **nella versione iniziale** del dataset vengono rimosse
con `drop_duplicates()` (da 41.188 a ~41.176 istanze). Si tratta di una scelta progettuale:
le operazioni di pulizia successive (in particolare l'imputazione degli `unknown`) generano
**nuovi** duplicati — circa 2.097 — che però **non vengono rimossi**, perché riflettono
osservazioni valide e coerenti col dominio e non distorcono né l'analisi né l'addestramento.

### 4.2 Distribuzione della classe

Si verifica lo **sbilanciamento** del target: la classe `yes` è circa l'**11%** del
totale. Questa consapevolezza è la premessa per la **stratificazione**
che verrà applicata negli split dei Task 4 e 5, e per la scelta delle metriche
(precision/recall/F1 invece della sola accuratezza).

### 4.3 Valori `unknown`

I valori mancanti sono codificati come la **stringa `"unknown"`** (lo dichiara la
documentazione UCI), non come celle vuote: per questo pandas non li vede come `NaN`
e UCI riporta formalmente "nessun valore mancante", anche se di fatto mancano. Sono
presenti in sei attributi nominali, con incidenza molto diversa: `default` (~20.9%),
`education` (~4.2%), `housing` e `loan` (~2.4%), `job` (~0.8%), `marital` (~0.2%).

**Scelta:** gli `unknown` vengono sostituiti con `NaN` e quindi **imputati con la moda**
(il valore più frequente) di ciascun attributo nominale, già in questo task. È una scelta
semplice e adatta al fatto che si tratta di variabili categoriche; l'imputazione avviene
sull'intero dataset prima dello split (lo split train/test vero e proprio è nei Task 4 e 5).

### 4.4 Rimozione di `duration` (data leakage)

La feature `duration` (durata in secondi dell'ultima telefonata) è **direttamente legata
al target**: una chiamata lunga è *conseguenza* dell'interesse del cliente, non una sua causa
predittiva, e il suo valore è noto **solo dopo** la chiamata. Usarla per predire la
sottoscrizione equivale a usare informazione proveniente dal futuro: è un caso di **data
leakage**. Per questo viene **rimossa già qui**, in fase di preparazione, con
`drop(columns=["duration"])`: il dataset passa così da 21 a **20 colonne** (19 feature + `y`)
e tutti i task successivi lavorano su dati privi di leakage. Da qui in avanti `duration` non
compare più — né nell'EDA del Task 3, né nella valutazione del Task 4, né nel modeling del Task 5.

---

## 5. Codifica della classe

Un classificatore matematico non interpreta le stringhe. La **classe** `y` viene
codificata con `yes → 1`, `no → 0`. La classe `yes` (cliente che sottoscrive il
deposito) è la classe **positiva**, quella che si vuole predire: per convenzione
vale 1.

La codifica degli **attributi nominali in ingresso** (one-hot, Lezione 11) viene
invece rimandata al Task 5, dentro una pipeline, per due motivi: dipende dal modello
scelto, e l'EDA del Task 3 è più leggibile sugli attributi nominali originali.

---

## 6. Estrazione dei due file

### 6.1 `training.csv`

**Scelta: si usa tutto il dataset pulito** (~41.176 istanze). Motivazioni:

- la classe `yes` è **rara**: ogni istanza positiva è preziosa, sottocampionare
  ridurrebbe il segnale già scarso;
- il volume è gestibile per i modelli previsti;
- la separazione train/test vera e propria si fa nei Task 4 e 5 con
  `train_test_split` (holdout stratificato).



### 6.2 `manuale.csv`

Serve per i calcoli **a mano** del Task 2. Criteri di costruzione:

- **mix di attributi nominali e numerici** — così il Task 2 può illustrare la gestione
  dei due tipi di attributo: la *discretizzazione sulla mediana* per gli attributi numerici
  in 1R e le *frequenze condizionate con stimatore di Laplace* per Naïve Bayes. Le feature
  selezionate sono `age`, `campaign` (numerici) e `job`, `marital`, `education`, `housing`,
  `loan`, `contact`, `poutcome` (nominali), più la classe `y` — **9 feature + target**;
- **poche istanze** — 12, per rendere fattibili i conti a
  mano;
- **set bilanciato** — 6 istanze `yes` e 6 `no` campionate con `random_state=SEED`:
  con soli `no` i calcoli sarebbero poco significativi. Le righe vengono infine
  mescolate (`sample(frac=1)`) così le classi non risultano raggruppate.

---

## 7. Output e passo successivo

Il task produce:

- **`training.csv`** — dataset pulito completo, pronto per EDA e addestramento;
- **`manuale.csv`** — 12 istanze bilanciate con attributi adatti ai calcoli a mano.

**Prossimo passo (Task 2):** definire a mano i due classificatori (1R e Naïve Bayes)
su `manuale.csv`, illustrarne i passi, implementarli in Python e valutarli.

---


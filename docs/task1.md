# Task 1 — Preparazione del dataset (Bank Marketing)

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** Bank Marketing (UCI) — `bank-additional-full.csv`
**Codice di riferimento:** `src/task1.ipynb`
**Libro di testo:** Witten, Frank, Hall, Pal — *Data Mining* (4ª ed.), cap. 4

---

## 1. Obiettivo del task

Trasformare il dataset grezzo in un formato comprensibile dai classificatori ed
estrarre due file di lavoro, usati nei task successivi:

| File | Dimensione | Scopo |
|------|-----------|-------|
| `manuale.csv` | 12 istanze | calcoli **a mano** dei classificatori (Task 2) |
| `training.csv` | dataset completo pulito | EDA e addestramento (Task 3, 4, 5) |

**Riferimenti del corso:** Lezione 3 (input del ML: attributi *nominali* vs
*numerici*, *istanze*), Lezione 5 (1R e Naïve Bayes, i due classificatori scelti),
Lezione 11 (feature engineering: one-hot encoding, imputazione dei mancanti).

---

## 2. Riproducibilità

Si fissa il seme dei generatori casuali (`SEED = 10`, `np.random.seed(SEED)`), come
nei notebook del corso. Questo garantisce che i file estratti — in particolare il
campionamento di `manuale.csv` — siano **sempre gli stessi** a ogni esecuzione e che
i risultati siano replicabili, requisito fondamentale per un progetto valutato.

---

## 3. Caricamento

Il file usa il **punto e virgola** (`;`) come separatore, come gli altri CSV del
corso; senza specificarlo pandas leggerebbe tutto in un'unica colonna. Il percorso
assume la struttura del repository (dataset in `../data/raw/`). Il dataset originale
conta 41.188 istanze × 21 colonne (20 attributi + classe `y`).

---

## 4. Controllo qualità di base

### 4.1 Duplicati

Le istanze identiche su tutti gli attributi non aggiungono informazione e
**distorcono le frequenze** che Naïve Bayes usa per stimare le probabilità. Vengono
quindi rimosse con `drop_duplicates()`. Dopo la pulizia restano ~41.176 istanze.

### 4.2 Distribuzione della classe

Si verifica lo **sbilanciamento** del target: la classe `yes` è circa l'**11%** del
totale. Questa consapevolezza è la premessa per la **stratificazione** (Lezione 8)
che verrà applicata negli split dei Task 4 e 5, e per la scelta delle metriche
(precision/recall/F1 invece della sola accuratezza).

### 4.3 Valori `unknown`

I valori mancanti sono codificati come la **stringa `"unknown"`** (lo dichiara la
documentazione UCI), non come celle vuote: per questo pandas non li vede come `NaN`
e UCI riporta formalmente "nessun valore mancante", anche se di fatto mancano. Il
notebook ne fa il censimento per colonna nominale, in modo vettoriale
(`(df[nominali] == "unknown").sum()`).

**Scelta:** per ora si mantengono come **categoria a sé**. L'eventuale imputazione
(Lezione 11, `SimpleImputer`) viene rimandata al Task 5 e applicata **solo** al
training set, dentro una pipeline, per evitare *data leakage*.

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
  `train_test_split` (holdout stratificato, Lezione 8).

Il file viene salvato **pulito ma ancora leggibile**: nominali come stringhe, classe
già 0/1.

### 6.2 `manuale.csv`

Serve per i calcoli **a mano** del Task 2. Criteri di costruzione:

- **mix di attributi nominali e numerici** — così entrambi i classificatori del Task
  2 possono illustrare la gestione dei due tipi di attributo: la *discretizzazione*
  per 1R e la *distribuzione gaussiana* per Naïve Bayes. Le feature selezionate sono
  `age`, `campaign` (numerici) e `job`, `marital`, `education`, `housing`, `loan`,
  `contact`, `poutcome` (nominali), più la classe `y`;
- **poche istanze** — 12, nel range 10–15 richiesto, per rendere fattibili i conti a
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

## 8. Concetti del corso utilizzati

| Concetto | Lezione |
|---|---|
| Attributi numerici vs nominali, istanze | 3 |
| 1R e Naïve Bayes (scelta dei classificatori) | 5 |
| Stratificazione e sbilanciamento delle classi | 8 |
| One-hot encoding, imputazione dei mancanti | 11 |
| Riproducibilità (seed) | trasversale |
| Data leakage (motiva il rinvio dell'imputazione) | trasversale |

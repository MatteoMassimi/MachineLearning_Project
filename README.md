# Bank Marketing — Progetto di Machine Learning

---

## Autori

- Matteo Massimi — 294140
- Lorenzo Coppolone — 292092

---

## Dataset

Si utilizza il dataset **Bank Marketing** dello UCI Machine Learning Repository,
nella versione arricchita `bank-additional-full.csv` (41.188 istanze, 20+ attributi classe), che include cinque variabili di contesto socioeconomico.

> ⚠️ **I dataset NON sono inclusi in questo repository** . Vanno scaricati e posizionati manualmente — vedi la sezione
> [Configurazione](#configurazione).

Fonte ufficiale: https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## Struttura del progetto

```
MachineLearning_Project/
├── README.md
├── requirements.txt
├── .gitignore
├── traccia_progetto.pdf   # traccia ufficiale del progetto
├── data/
│   ├── raw/               # dataset originale (da scaricare, non incluso)
│   └── processed/         # manuale.csv e training.csv (generati dal Task 1)
├── docs/                  # documentazione: un file .md per ogni task
│   ├── task1.md           # preparazione del dataset
│   ├── task2.md           # classificatori manuali (1R e Naïve Bayes)
│   ├── task3.md           # analisi esplorativa (EDA)
│   ├── task4.md           # valutazione dei classificatori manuali
│   └── task5.md           # addestramento con Scikit-Learn
└── src/
    ├── task1.ipynb               # preparazione del dataset
    ├── task2_1R.ipynb            # classificatore manuale: 1R
    ├── task2_NaiveBayes.ipynb    # classificatore manuale: Naïve Bayes
    ├── task3.ipynb               # analisi esplorativa (EDA)
    ├── task4.ipynb               # valutazione dei classificatori manuali
    └── task5.ipynb               # addestramento con Scikit-Learn
```

> **Documentazione.** Il codice vive nei notebook in `src/`; la **documentazione
> discorsiva** di ogni task (cosa è stato fatto e perché) è raccolta in `docs/`,
> con un file `.md` per task. Ogni notebook rimanda in cima al rispettivo documento.

---

## Configurazione

### 1. Scaricare il dataset

I dati grezzi non sono presenti nella cartella. Seguire i seguenti passi:

1. Aprire https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Scaricare l'archivio e scompattarlo
3. Creare un package data e al suo interno due sottopackage : raw (contiene il dataset grezzo e la relativa documentazione) e processed ( contiene i datasets ottenuti nel Task 1)
3. Copiare il file `bank-additional-full.csv` (e `bank-additional-names.txt`)
   nella cartella `data/raw/`

La struttura attesa dopo questo passo è:

```
data/raw/
├── bank-additional-full.csv
└── bank-additional-names.txt
```

### 2. Creare e attivare l'ambiente virtuale

L'ambiente virtuale (`.venv`) isola le librerie del progetto. **Non è incluso nel
repository** ed è ricreabile dai comandi seguenti.

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

A ambiente attivo, il prompt mostra `(.venv)` all'inizio della riga.

### 3. Installare le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Eseguire i task

Poiche i Tasks 2-5 utilizzano datasets generati nel Task 1 , si consiglia di eseguire i tasks **in ordine**.

Ordine consigliato:
1. `task1.ipynb` — genera `manuale.csv` e `training.csv`
2. `task2_1R.ipynb` e `task2_NaiveBayes.ipynb` — classificatori manuali
3. `task3.ipynb` — analisi esplorativa (EDA)
4. `task4.ipynb` — valutazione dei classificatori manuali
5. `task5.ipynb` — addestramento con Scikit-Learn

---

## Tasks del progetto

1. **Preparazione del dataset** — pulizia ed estrazione di `manuale.csv` (12 istanze
   per i calcoli a mano) e `training.csv` (dataset di lavoro).
2. **Classificatori manuali** — definizione e implementazione a mano di due modelli
   (1R e Naïve Bayes) su `manuale.csv`, in due notebook distinti.
3. **Analisi esplorativa (EDA)** — controllo qualità, boxplot, pairplot, matrice di
   correlazione su `training.csv`.
4. **Valutazione dei classificatori manuali** su `training.csv` e ottimizzazione.
5. **Addestramento con Scikit-Learn** — più classificatori, holdout stratificato,
   ottimizzazione degli iperparametri, selezione del modello migliore.


# Bank Marketing — Progetto di Machine Learning

Progetto per il corso **Fondamenti e Applicazioni del Machine Learning (FML 2026)**.

L'obiettivo è costruire un classificatore binario che predica se un cliente di una
banca portoghese **sottoscriverà un deposito a termine** (`y = yes/no`), a partire
dai dati di una campagna di telemarketing.

---

## Dataset

Si utilizza il dataset **Bank Marketing** dello UCI Machine Learning Repository,
nella versione arricchita `bank-additional-full.csv` (41.188 istanze, 20 attributi
+ classe), che include cinque variabili di contesto socioeconomico.

> ⚠️ **I dataset NON sono inclusi in questo repository** (sono esclusi tramite
> `.gitignore`). Vanno scaricati e posizionati manualmente — vedi la sezione
> [Configurazione](#configurazione).

Fonte ufficiale: https://archive.ics.uci.edu/dataset/222/bank+marketing

---

## Struttura del progetto

```
bank-marketing-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # dataset originale (da scaricare, non incluso)
│   └── processed/      # manuale.csv e training.csv (generati dal Task 1)
├── notebooks/
│   ├── task1_preparazione.ipynb
│   ├── task2_classificatori_manuali.ipynb
│   ├── task3_eda.ipynb
│   ├── task4_valutazione_manuali.ipynb
│   └── task5_sklearn.ipynb
├── results/            # grafici e output generati
└── docs/
    └── documentazione.pdf
```

---

## Configurazione

### 1. Clonare il repository

```bash
git clone https://github.com/<utente>/bank-marketing-ml.git
cd bank-marketing-ml
```

### 2. Scaricare il dataset

I dati non sono versionati. Scaricarli dallo UCI Repository:

1. Aprire https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Scaricare l'archivio e scompattarlo
3. Copiare il file `bank-additional-full.csv` (e `bank-additional-names.txt`)
   nella cartella `data/raw/`

La struttura attesa dopo questo passo è:

```
data/raw/
├── bank-additional-full.csv
└── bank-additional-names.txt
```

### 3. Creare e attivare l'ambiente virtuale

L'ambiente virtuale (`venv`) isola le librerie del progetto. **Non è incluso nel
repository** ed è ricreabile dai comandi seguenti.

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

A ambiente attivo, il prompt mostra `(venv)` all'inizio della riga.

### 4. Installare le dipendenze

```bash
pip install -r requirements.txt
```

### 5. Eseguire i task

I task sono notebook Jupyter, da eseguire **in ordine** (il Task 1 genera i file
usati dagli altri). Avviare Jupyter e aprire i notebook nella cartella `src/`:

```bash
jupyter src
```

In alternativa, da riga di comando si può eseguire un notebook senza aprirlo:

```bash
jupyter nbconvert --to src --execute --inplace src/task1.ipynb
```

Ordine consigliato:
1. `task1.ipynb` — genera `manuale.csv` e `training.csv`
2. `task2_classificatori_manuali.ipynb`
3. `task3_eda.ipynb`
4. `task4_valutazione_manuali.ipynb`
5. `task5_sklearn.ipynb`

---

## Task del progetto

1. **Preparazione del dataset** — pulizia ed estrazione di `manuale.csv` (12 istanze
   per i calcoli a mano) e `training.csv` (dataset di lavoro).
2. **Classificatori manuali** — definizione e implementazione a mano di due modelli
   (Naïve Bayes e KNN) su `manuale.csv`.
3. **Analisi esplorativa (EDA)** — controllo qualità, boxplot, pairplot, matrice di
   correlazione su `training.csv`.
4. **Valutazione dei classificatori manuali** su `training.csv` e ottimizzazione.
5. **Addestramento con Scikit-Learn** — più classificatori, holdout stratificato,
   ottimizzazione degli iperparametri, selezione del modello migliore.

---

## Autori

- Nome Cognome — matricola
- Nome Cognome — matricola

---

## Note

- Tutti gli script fissano il seme dei generatori casuali per garantire la
  **riproducibilità** dei risultati.
- L'attributo `duration` viene escluso dalla modellazione finale: non è noto prima
  della telefonata, quindi includerlo produrrebbe un modello non realistico.
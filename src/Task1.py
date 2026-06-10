
"""
==================================================================
 TASK 1 - Preparazione del dataset (Bank Marketing)
 Corso: Fondamenti e Applicazioni del Machine Learning (FML 2026)
==================================================================
Riferimenti del corso utilizzati:
  - Libro di testo: Witten, Frank, Hall, Pal, "Data Mining" (4a ed.), cap. 4
  - Lezione 3  (Input del ML): attributi nominali/categorici vs numerici, istanze
  - Lezione 11 (Feature Engineering): one-hot encoding (OneHotEncoder),
                imputazione dei mancanti (SimpleImputer)
  - Lezioni 5-7 (Algoritmi di base): 1R, Naive Bayes (con stimatore di Laplace
                e ipotesi gaussiana per i numerici), alberi decisionali,
                modelli lineari, instance-based (KNN, distanza euclidea)
 
Obiettivo del Task 1:
  1) Caricare correttamente il dataset grezzo
  2) Controllo qualita' di base (duplicati, target, valori 'unknown')
  3) Trasformarlo in un formato comprensibile dal classificatore
  4) Estrarre:
        - manuale.csv  -> 10-15 istanze, per i calcoli a mano (Task 2)
        - training.csv -> dataset di lavoro pulito (Task 3, 4, 5)
 
Eseguire con:  python3 task1_preparazione.py
==================================================================
"""
 
import pandas as pd
import numpy as np
 
# Riproducibilita': nei notebook del corso il seme dei generatori casuali
# viene fissato con np.random.seed(...) (es. valori 10, 12) per ottenere
# risultati replicabili. Lo fissiamo anche noi: garantisce che i file
# estratti siano sempre gli stessi e che i risultati siano verificabili.
SEED = 10
np.random.seed(SEED)
 
# ------------------------------------------------------------------
# STEP 1 - CARICAMENTO
# ------------------------------------------------------------------
# Il file usa il PUNTO E VIRGOLA come separatore (come gli altri CSV
# usati nel corso, es. elenco-comuni.csv letto con sep=';').
# Senza specificarlo, pandas leggerebbe tutto in un'unica colonna.
PATH = "Datasets/bank-additional-full.csv"
dataSet = pd.read_csv(PATH, sep=";")
 
print(f"[STEP 1] Caricato dataset: {dataSet.shape[0]} istanze x {dataSet.shape[1]} attributi")
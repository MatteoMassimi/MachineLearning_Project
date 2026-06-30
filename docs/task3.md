# Task 3 — Analisi esplorativa di `training.csv`

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** `training.csv` (~41.176 istanze × 20 attributi, generato dal Task 1)
**Codice di riferimento:** `src/task3.ipynb`

---

## 1. Obiettivo del task


1. **verificare** che il dataset non contenga osservazioni palesemente errate;
2. effettuare l'**analisi esplorativa** (EDA), rappresentando i risultati anche in
   forma grafica: **boxplot** e **matrice di correlazione**.

> **Nota su `duration`.** A differenza di altre impostazioni del problema, qui `duration`
> **non compare**: è stato **rimosso nel Task 1** perché costituisce *data leakage* (la durata
> della chiamata è nota solo *dopo* la chiamata). L'EDA lavora quindi sui 19 attributi predittivi
> realmente utilizzabili, più il target.

---

## 2. Verifica di osservazioni errate

### 2.1 Statistiche descrittive e valori impossibili

**Nessun valore impossibile**: tutti i minimi sono ≥ 0 (nessuna età negativa; l'età va da 17 a 98
anni). Emergono però due elementi da interpretare: `pdays` arriva a 999 e `campaign` ha un massimo
molto alto (56).

### 2.2 Il codice speciale di `pdays`

`pdays` (giorni dall'ultimo contatto di una campagna precedente) vale **999** come
**codice convenzionale** per "mai contattato prima", non come valore reale. Risulta
che il **~96%** (precisamente 96.3%) dei clienti non era mai stato contattato: `pdays` è quindi
una **near-constant feature**, poco informativa e potenziale fonte di rumore. Per questo verrà
**escluso dalle variabili di addestramento** nel Task 5.

### 2.3 Valori mancanti (`unknown`)

I mancanti del dataset originale erano la stringa `"unknown"`. **Sono già stati imputati con la
moda nel Task 1**, quindi in `training.csv` non sono più presenti come categoria a sé.

### 2.4 Duplicati

Le istanze identiche **iniziali** sono già state rimosse nel Task 1. I duplicati generati
successivamente dall'imputazione sono stati **deliberatamente mantenuti** (osservazioni valide e
coerenti col dominio), come discusso nel Task 1: non rappresentano un problema per l'analisi.

### 2.5 Outlier estremi

`campaign` arriva a **56 contatti** nella stessa campagna: un valore **plausibile ma anomalo**,
non un errore da rimuovere. Viene annotato perché alcuni modelli sono sensibili agli outlier.

**Conclusione:** nessuna osservazione palesemente errata; gli unici punti da
*interpretare* sono il codice `pdays=999` e gli outlier di coda.

---

## 3. Boxplot degli attributi numerici

Il **boxplot** mostra mediana, quartili e outlier di ogni variabile.

Si osservano **scale molto diverse** (es. `nr.employed` ~5000 contro `previous`
~0–7): un motivo per cui alcuni modelli richiederanno la **standardizzazione** (Task
5). Si vedono inoltre marcate **code destre** (molti outlier) in `campaign` e `previous`, tipiche
di variabili di conteggio, mentre `pdays` appare schiacciato sul valore 999.

---

## 4. Matrice di correlazione


**Osservazioni critiche:**

- **Nessun predittore numerico è fortemente correlato con il target.** Le correlazioni con `y`
  sono tutte modeste: la più alta in valore assoluto è **`nr.employed` (−0.355)**, seguita da
  `pdays` (−0.325), `euribor3m` (−0.308), `emp.var.rate` (−0.298) e, con segno **positivo**,
  `previous` (+0.230). È un segnale importante: la sola informazione numerica non basterà, e gli
  **attributi nominali** avranno un ruolo determinante.
- Le **variabili economiche** (`nr.employed`, `euribor3m`, `emp.var.rate`, `pdays`)
  sono **negativamente** correlate con `y`: quando l'economia "tira" (più occupati,
  tassi alti), i clienti sottoscrivono meno depositi.
- **Multicollinearità:** `emp.var.rate`, `euribor3m` e `nr.employed` sono fortemente
  correlate **tra loro** (>0.90). Sono quasi ridondanti: alcuni modelli (es. la
  regressione logistica) ne risentono, e Naïve Bayes vede violata l'assunzione di
  indipendenza. Punto da riprendere nel Task 5.

---


## 5. Analisi degli attributi nominali

Per le variabili nominali l'indicatore più utile è il **tasso di sottoscrizione**
(media di `y`) per categoria, visualizzato con barplot orizzontali per `job`, `education`,
`poutcome` e `month`.

**Osservazioni:**

- **`poutcome=success`** ha un tasso di sottoscrizione del **~65%** (esattamente 0.651, contro
  l'~11% medio): chi ha già aderito a una campagna precedente è molto propenso a riaderire.
  È probabilmente l'attributo nominale più predittivo (`failure` 0.142, `nonexistent` 0.088).
- Alcuni mesi hanno tassi molto più alti della media: forte **stagionalità**.
- Anche `job` ed `education` mostrano categorie con tassi nettamente sopra/sotto la media.

---

## 5. Sintesi e passo successivo

- **Nessuna osservazione palesemente errata**; da interpretare solo `pdays=999` e gli outlier di coda.
- **Nessun predittore numerico forte**: la massima correlazione con `y` è ~0.36 (`nr.employed`).
- Forte **multicollinearità** tra le variabili economiche.
- Attributi nominali molto informativi: **`poutcome`**, `month`, e in misura minore `job`/`education`.

**Prossimo passo (Task 4):** valutare i classificatori manuali (1R e Naïve Bayes) su
`training.csv` e discuterne i margini di ottimizzazione.

---

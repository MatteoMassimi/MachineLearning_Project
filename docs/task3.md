# Task 3 — Analisi esplorativa di `training.csv`

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** `training.csv` (~41.176 istanze, generato dal Task 1)
**Codice di riferimento:** `src/task3.ipynb`

---

## 1. Obiettivo del task

La traccia chiede di:

1. **verificare** che il dataset non contenga osservazioni palesemente errate;
2. effettuare l'**analisi esplorativa** (EDA), rappresentando i risultati anche in
   forma grafica: **boxplot**, **pairplot** e **matrice di correlazione**.

**Stile di codice.** Vettoriale (pandas/numpy/`groupby`), senza cicli `for`
espliciti. I grafici usano **matplotlib** (stile `seaborn-v0_8`, come nei notebook
del corso) e **seaborn** per heatmap e pairplot. Gli attributi numerici e nominali
sono separati con `select_dtypes` (Lezione 3).

> **Nota su `duration`.** Viene **incluso** nell'analisi proprio per *vedere* il suo
> effetto, pur sapendo (documentazione UCI) che la durata della chiamata non è nota
> *prima* della chiamata e andrà quindi **esclusa** dal modello predittivo (Task 5).

---

## 2. Verifica di osservazioni errate

### 2.1 Statistiche descrittive e valori impossibili

`describe()` riassume le numeriche. **Nessun valore impossibile**: tutti i minimi
sono ≥ 0 (nessuna età o durata negativa). Emergono però due elementi da
interpretare: `pdays` arriva a 999 e `duration`/`campaign` hanno massimi molto alti.

### 2.2 Il codice speciale di `pdays`

`pdays` (giorni dall'ultimo contatto di una campagna precedente) vale **999** come
**codice convenzionale** per "mai contattato prima", non come valore reale. Risulta
che **~96%** dei clienti non era mai stato contattato: `pdays` così com'è è quindi
poco informativo e andrebbe trasformato (nota per il Task 5).

### 2.3 Valori mancanti (`unknown`)

Come nel Task 1, i mancanti sono la stringa `"unknown"`. Vengono contati per attributo
nominale (vettoriale con `apply`) per quantificarne la diffusione.

### 2.4 Duplicati

Nessun duplicato residuo: erano già stati rimossi nel Task 1.

### 2.5 Outlier estremi

`campaign` arriva a 56 contatti nella stessa campagna e `duration` a oltre 80 minuti.
Sono valori **plausibili ma anomali**, non errori da rimuovere. Vengono annotati
perché alcuni modelli sono sensibili agli outlier.

**Conclusione:** nessuna osservazione palesemente errata; gli unici punti da
*interpretare* sono il codice `pdays=999` e i valori `unknown`.

---

## 3. Boxplot degli attributi numerici

Il **boxplot** mostra mediana, quartili e outlier di ogni variabile. Si usa
`plot(kind="box", subplots=True)` per generare una griglia senza ciclo `for`.

Si osservano **scale molto diverse** (es. `nr.employed` ~5000 contro `previous`
~0–7): un motivo per cui alcuni modelli richiederanno la **normalizzazione** (Task
5). Si vede inoltre la forte presenza di **outlier** in `duration` e `campaign`.

---

## 4. Matrice di correlazione

Calcolata con `df.corr()` e visualizzata con una **heatmap** di seaborn. Le
correlazioni con il target vengono poi ordinate.

**Osservazioni critiche (centrali per l'orale):**

- **`duration` è il più correlato con il target (~0.41)**, di gran lunga. Ma è
  proprio l'attributo da **escludere**: la durata è nota solo *dopo* la chiamata,
  quando l'esito è già deciso. Questa correlazione alta è esattamente il **data
  leakage** — spettacolare sui dati storici, inutile nella realtà. Verrà escluso nel
  Task 5.
- Le **variabili economiche** (`nr.employed`, `euribor3m`, `emp.var.rate`, `pdays`)
  sono **negativamente** correlate con `y`: quando l'economia "tira" (più occupati,
  tassi alti), i clienti sottoscrivono meno depositi.
- **Multicollinearità:** `emp.var.rate`, `euribor3m` e `nr.employed` sono fortemente
  correlate **tra loro** (>0.90). Sono quasi ridondanti: alcuni modelli (es. la
  regressione logistica) ne risentono, e Naïve Bayes vede violata l'assunzione di
  indipendenza. Punto da riprendere nel Task 5.

---

## 5. Pairplot

Mostra le relazioni a coppie e le distribuzioni sulla diagonale, colorando per
classe. Viene fatto su un **sottoinsieme** di numeriche significative (`age`,
`duration`, `campaign`, `euribor3m`, `nr.employed`) e su un **campione** di 2000
istanze (su 41.000 punti il grafico sarebbe lento e sovraffollato).

La classe `yes` tende a concentrarsi a **`duration` elevata** e a **`euribor3m`
basso**, coerentemente con la matrice di correlazione.

---

## 6. Analisi degli attributi nominali

Per le variabili nominali l'indicatore più utile è il **tasso di sottoscrizione**
(media di `y`) per categoria, calcolato con `groupby` e visualizzato con barplot
orizzontali.

**Osservazioni:**

- **`poutcome=success`** ha un tasso di sottoscrizione del **~65%** (contro l'~11%
  medio): chi ha già aderito a una campagna precedente è molto propenso a riaderire.
  È probabilmente l'attributo nominale più predittivo.
- **`contact=cellular`** converte molto meglio di `telephone`.
- Alcuni mesi (marzo, dicembre, settembre, ottobre) hanno tassi molto più alti: forte
  **stagionalità**.

---

## 7. Sintesi e passo successivo

- **Nessuna osservazione palesemente errata**; da interpretare solo `pdays=999` e gli
  `unknown`.
- **`duration`** è il più correlato col target ma va **escluso** dal modello (data
  leakage).
- Forte **multicollinearità** tra le variabili economiche.
- Attributi nominali molto informativi: **`poutcome`**, `contact`, `month`.

**Prossimo passo (Task 4):** valutare i classificatori manuali (1R e Naïve Bayes) su
`training.csv` e ottimizzarne le prestazioni.

---

## 8. Concetti del corso utilizzati

| Concetto | Lezione |
|---|---|
| Attributi numerici vs nominali | 3 |
| EDA, boxplot, correlazione, pairplot | trasversale / 8 |
| Necessità di normalizzazione (scale diverse) | 11 |
| Data leakage (`duration`) | trasversale |
| Multicollinearità | trasversale |

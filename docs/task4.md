# Task 4 — Valutazione dei classificatori manuali su `training.csv`

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** `training.csv` (~41.176 istanze)
**Codice di riferimento:** `src/task4.ipynb`

---

## 1. Obiettivo del task

La traccia chiede di prendere i classificatori del Task 2 (**1R** e **Naïve Bayes**),
valutarli su `training.csv` (o un suo sottoinsieme) e **ottimizzarne** le prestazioni.

Rispetto al Task 2 (12 istanze, leave-one-out) qui si lavora su **41.176 istanze**,
quindi:

- si usa un **holdout stratificato** (Lezione 8): 70% training, 30% test, mantenendo
  la proporzione delle classi (`stratify=y`, `random_state=10`);
- si usano metriche adatte a una **classe sbilanciata** (~11% di `yes`): non solo
  accuratezza, ma **precision, recall, F1** e **matrice di confusione**.

Si mostra anche l'**effetto di `duration`** (con e senza), per quantificare il *data
leakage* discusso nel Task 3. **Stile di codice** vettoriale, senza cicli `for`.

---

## 2. Il baseline da battere

Con classi sbilanciate, un classificatore "stupido" che predice **sempre `no`**
ottiene già un'accuratezza di **~89%**, ma **recall su `yes` = 0%**. È il
**riferimento minimo**: un modello utile deve fare meglio di così, soprattutto sul
recall della classe `yes`. Questo dimostra subito perché l'accuratezza, da sola, è
una metrica fuorviante su questo dataset.

---

## 3. Naïve Bayes su `training.csv`

Si usa `GaussianNB` di sklearn come controprova del Task 2. Gli attributi nominali
vengono codificati con `OrdinalEncoder` (con `handle_unknown="use_encoded_value"`,
`unknown_value=-1`); l'encoder è **fittato sul solo training** e applicato al test,
così da evitare leakage. Una funzione `valuta_nb(usa_duration)` addestra e valuta con
o senza `duration`.

**Osservazioni:**

- **Accuratezza sotto il baseline** (~84% vs ~89%): Naïve Bayes "sacrifica" un po' di
  accuratezza globale, ma in cambio **individua molti `yes`** (recall ~50–60%) che il
  baseline ignorerebbe. Per una campagna di marketing è proprio ciò che serve:
  trovare i clienti propensi.
- **Effetto di `duration`:** includerlo alza F1 e recall — è il **data leakage**
  quantificato. Il guadagno è illusorio: nella realtà `duration` non è disponibile
  prima della chiamata. Per questo verrà escluso nel Task 5.

---

## 4. 1R su `training.csv`

Si riusa l'idea del Task 2 (errori via `crosstab`), adattata al dataset grande. Per
gli attributi numerici si discretizza con `pd.qcut` (intervalli a uguale frequenza).
**Ottimizzazione:** si provano diversi numeri di intervalli (`bins` = 3, 5, 10).

### 4.1 Quale attributo sceglie 1R?

Calcolando gli errori di ogni attributo (numerici discretizzati in 5 bin), si osserva
che **quasi tutti** fanno esattamente `n_yes` errori: la loro regola 1R degenera nel
**predire sempre `no`**. Solo **`poutcome`** e **`month`** fanno meglio del baseline.
È un limite strutturale di 1R su dati sbilanciati: con un solo attributo è difficile
catturare la classe rara.

### 4.2 Ottimizzazione dei bin

Il numero di bin **non cambia** il risultato, perché 1R sceglie comunque `poutcome`
(nominale, non soggetto a binning). L'ottimizzazione dei bin è quindi ininfluente:
risultato di per sé interessante da spiegare.

---

## 5. Confronto finale 1R vs Naïve Bayes

Confronto nello scenario realistico (**senza `duration`**) contro il baseline, su
recall e F1 (le metriche che contano su classe sbilanciata):

- **1R** ottiene un'**accuratezza alta** (sceglie `poutcome`, batte di poco il
  baseline) ma un **recall molto basso**: cattura pochi `yes`. È trasparente e
  semplice, ma il suo singolo attributo non basta su dati sbilanciati.
- **Naïve Bayes** ha accuratezza più bassa ma **recall nettamente migliore**: usando
  tutti gli attributi individua molti più clienti propensi. Per l'obiettivo di
  business (marketing) è preferibile.

---

## 6. Analisi critica

**Sull'ottimizzazione.** Per 1R provare diversi numeri di bin è risultato
ininfluente, perché vince comunque un attributo nominale (`poutcome`). L'accuratezza
**non** è la metrica giusta qui: il baseline al ~89% lo dimostra; **F1 e recall**
sulla classe `yes` sono molto più informativi.

**Sull'effetto `duration`.** Includerlo migliora le metriche di entrambi i modelli,
ma è un guadagno **illusorio** (*data leakage*): non sarà disponibile in un contesto
reale. Conferma la decisione di **escluderlo** nel Task 5.

---

## 7. Passo successivo

**Task 5:** addestrare più classificatori con Scikit-Learn, con preprocessing
completo (one-hot, scaling, imputazione in pipeline), ottimizzazione degli
iperparametri e selezione del modello migliore sul test set.

---



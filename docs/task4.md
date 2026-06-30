# Task 4 — Valutazione dei classificatori manuali su `training.csv`

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** `training.csv` (~41.176 istanze × 20 attributi)
**Codice di riferimento:** `src/task4_1R.ipynb` e `src/task4_NaiveBayes.ipynb`

---

## 1. Obiettivo del task

La traccia chiede di prendere i classificatori del Task 2 (**1R** e **Naïve Bayes**),
valutarli su `training.csv` e **ottimizzarne** le prestazioni. Il lavoro è diviso in **due
notebook distinti**, uno per classificatore.

**Approccio adottato.** I due modelli **non vengono riaddestrati**: si riprendono *tal quali* dal
Task 2 (la regola e le probabilità apprese sulle 12 istanze di `manuale.csv`) e si applicano
all'**intero `training.csv`** (~41.176 istanze). Non c'è quindi un nuovo `train_test_split`: il
file viene usato interamente come banco di prova per i modelli costruiti a mano. Si usano metriche
adatte a una **classe sbilanciata** (~11% di `yes`): non solo accuratezza, ma **precision, recall,
F1** e **matrice di confusione**. **Stile di codice** vettoriale, senza cicli `for`.

> **Nota.** `duration` è già stato rimosso nel Task 1, quindi qui non c'è alcun confronto
> "con/senza `duration`": il *data leakage* è stato eliminato a monte.

---

## 2. 1R su `training.csv` (`task4_1R.ipynb`)

Si riprende il **modello 1R appreso nel Task 2** (attributo **`marital`**, regola
`{divorced → 1, married → 0, single → 1}`) e la funzione `predici_1R`, applicata riga per riga.

**Risultati (modello originale):**

| Metrica | Valore |
|---|---|
| Accuracy  | 59.63% |
| Precision | 12.96% |
| Recall    | 45.18% |
| F1-Score  | 20.14% |

Matrice di confusione: TN=22.458, FP=14.079, FN=2.543, TP=2.096.

**Osservazioni.** La regola `marital`, appresa su un file bilanciato, assegna alla classe `1` i
clienti `single` e `divorced`: sul dataset reale (dove i `no` dominano) questo produce
**moltissimi falsi positivi** (14.079), da cui una precision molto bassa (12.96%). La recall
(45.18%) mostra che il modello individua meno della metà dei reali positivi, e l'F1 (20.14%)
conferma un compromesso tutt'altro che ottimale.

**Perché 1R non è ottimizzabile senza riaddestramento.** Il "modello" 1R è semplicemente una
**regola di maggioranza**: non ha probabilità a priori, soglie o iperparametri da regolare a
posteriori. L'unica cosa modificabile sarebbe la mappa valore → classe, ma ricavarla significa
**riaddestrare** la regola sui dati — operazione che, in coerenza con l'approccio del Task 2, qui
si è scelto di non fare. Per questo nel notebook **non compare alcuna fase di ottimizzazione**: è
di per sé un risultato significativo, perché mostra come la **semplicità** di 1R (un pregio nel
Task 2) si traduca qui in **rigidità**.

---

## 3. Naïve Bayes su `training.csv` (`task4_NaiveBayes.ipynb`)

Si riprendono le **probabilità condizionate** del Task 2 (sulle tre feature `marital`, `housing`,
`loan`, con Laplace) e la funzione `predici_naive_bayes`, applicata all'intero `training.csv`.

**Risultati (modello originale, priori 0.5 / 0.5):** sono **identici** a quelli di 1R
(Accuracy 59.63%, Precision 12.96%, Recall 45.18%, F1 20.14%, stessa matrice di confusione).
Il perché è spiegato nella Sezione 6.

### 4.1 L'unica leva disponibile: le probabilità a priori

Non riaddestrando il modello, l'unico parametro modificabile è il **prior**. Si prova quindi a
sostituire i priori bilanciati (0.5 / 0.5) con quelli **reali** del dataset, stimati nel Task 1
(**≈ 0.887 / 0.113**), tramite la funzione `predici_naive_bayes_ottimizzato`.

**Risultati (modello "ottimizzato"):**

| Metrica | Modello originale | Modello "ottimizzato" |
|---|:---:|:---:|
| Accuracy  | 59.63% | **88.73%** |
| Precision | 12.96% | 0% |
| Recall    | 45.18% | 0% |
| F1-Score  | 20.14% | 0% |

Matrice di confusione del modello ottimizzato: TN=36.537, FP=0, **FN=4.639, TP=0**.

**È un miglioramento illusorio — il paradosso dell'accuratezza.** Con un prior della classe `0`
così dominante, le tre verosimiglianze (deboli) non riescono **mai** a ribaltarlo: lo `score_1`
non supera mai lo `score_0`, e il modello **degenera** predicendo `0` per *tutte* le 41.176
istanze. L'88.73% di accuratezza coincide esattamente con la **proporzione della classe
maggioritaria** — cioè con ciò che farebbe un banale *Zero-R* ("sempre no"). Precision, recall e
F1 a **zero** rivelano l'inutilità del modello: non individua **nessuno** dei 4.639 clienti che
hanno effettivamente sottoscritto.

---

## 4. Analisi critica dell'ottimizzazione

Intervenire solo sulle probabilità a priori **non è una vera ottimizzazione**: sposta il modello
lungo il trade-off precision/recall senza migliorarne il potere discriminante. Si ottengono solo
i due estremi — o troppi falsi positivi (priori bilanciati), o nessun positivo (priori reali) —
anche perché il classificatore poggia su appena **tre feature**, due delle quali (`housing`,
`loan`) già rivelatesi **poco discriminanti** nel Task 2.

Le leve realmente efficaci — **riaddestrare** i modelli su `training.csv`, introdurre **attributi
più informativi** (es. `poutcome`, `month`, emersi nel Task 3), **discretizzare** le variabili
numeriche, e soprattutto **gestire esplicitamente lo sbilanciamento** (pesi di classe o
ricampionamento) — esulano dalla logica "a mano" di questo task e vengono affrontate nel **Task 5**.

---

## 5. Perché 1R e Naïve Bayes danno gli stessi risultati

I due classificatori producono **esattamente le stesse metriche originali**. **Non è una
coincidenza**, ma una conseguenza diretta delle feature scelte nel Task 2. La decisione di Naïve
Bayes è $\arg\max_c\ P(c)\cdot P(\text{marital}\mid c)\cdot P(\text{housing}\mid c)\cdot P(\text{loan}\mid c)$, e in questo confronto:

- i **priori** sono uguali (0.5 / 0.5) e si **elidono**;
- **`housing`** ha distribuzioni **identiche** nelle due classi: il suo fattore è lo stesso per
  `score_0` e `score_1` e si **semplifica**;
- **`loan`** è troppo debole: il suo contributo non basta mai a ribaltare la classe suggerita da `marital`.

Di fatto la predizione è guidata dal **solo `marital`**, esattamente l'attributo su cui 1R
costruisce la propria regola: ecco perché i due modelli prendono le **stesse** decisioni. È un
risultato **atteso e spiegabile**, non un errore: mostra come un Naïve Bayes su poche feature
poco discriminanti **degeneri** in un classificatore a regola singola, indistinguibile da 1R.

---

## 6. Passo successivo

**Task 5:** addestrare più classificatori con Scikit-Learn, con preprocessing
completo (one-hot, scaling in pipeline), gestione dello sbilanciamento (`class_weight`),
ottimizzazione degli iperparametri e selezione del modello migliore sul test set. È lì che la
differenza tra i due approcci potrà essere valutata in modo statisticamente fondato.

---

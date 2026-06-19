# Task 2 — Classificatori manuali: 1R e Naïve Bayes

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** `manuale.csv` (12 istanze, generato dal Task 1)
**Codice di riferimento:** `src/task2_1R.ipynb` e `src/task2_NaiveBayes.ipynb`


---

## 1. Obiettivo del task
L'Obiettivo e quello di **definire, adattare ai dati,
implementare e valutare a mano due classificatori** su `manuale.csv`. I due scelti
sono **1R** e **Naïve Bayes**, separati in **due notebook distinti** per chiarezza.



**Valutazione sullo stesso file.** 
Entrambi iclassificatori vengono valutati **sullo stesso `manuale.csv`** su cui sono costruiti .
I risultanti ottenuti non sono da considerare rilevanti qualitativamente in quanti abbiamo lavorato appunto su un dataset giocattolo :
 la valutazione statisticamente robusta è neiTask 4 e 5.



**Attributi:** numerici `age`, `campaign`; nominali `job`, `marital`, `education`,
`housing`, `loan`, `contact`, `poutcome`. Le 12 istanze sono bilanciate 6 `yes` / 6 `no`.

---

## 2. Classificatore 1R (1-Rule)

### 2.1 Teoria

1R costruisce un **albero decisionale a un solo livello**: classifica usando un unico
attributo. La procedura:

1. **per ogni attributo** si genera una regola: per ciascun valore dell'attributo si
   guarda quale **classe è più frequente** tra le istanze con quel valore e si
   assegna quella classe;
2. si **contano gli errori** della regola sul training set;
3. si **sceglie l'attributo** la cui regola produce il **minor numero di errori**.

Per gli **attributi numerici** (`age`, `campaign`) 1R richiede una
**discretizzazione**: i valori continui vengono divisi in intervalli (*bin*). Nel
notebook si usa un binning semplice sulla **mediana** (due intervalli).

> **Punto critico :** 1R tende all'**overfitting** quando un attributo
> ha **molti valori distinti**. Al limite, un valore diverso per ogni istanza darebbe
> 0 errori ma sarebbe inutile.

### 2.2 Adattamento ai dati e implementazione

- Inizialmente sono state reailzzate due funzioni : `regole_e_errori_attributo` che implementa 1R per un singolo attributo
 e `errori_nominale` che invece applica 1R a tutti gli attributi nominali
 
- Gli errori si calcolano per **tutti** gli attributi (nominali direttamente,
  numerici dopo binning sulla mediana), iterando con `apply` su una `Series`.
- `addestra_1R` sceglie l'attributo con meno errori e ne memorizza la regola;
  `predici_1R` applica la regola a una nuova istanza (con default `0` se incontra un
  valore mai visto).

### 2.3 Risultati e osservazioni critiche

Valutato **sullo stesso `manuale.csv`**, 1R sceglie l'attributo **`job`** e ottiene:

| Metrica | Valore |
|---|---|
| Accuracy  | 83.33% |
| Precision | 83.33% |
| Recall    | 83.33% |
| F1-Score  | 83.33% |

Matrice di confusione: TN=5, FP=1, FN=1, TP=5 (10 istanze su 12 corrette).

- **Pareggio tra `job` e `marital`** (entrambi 2 errori sul training): 1R sceglie il primo per
  *tie-breaking*. Tuttavia `job` ha **4 valori con una sola istanza**, ognuno dei quali dà 0
  errori "gratis": è l'**overfitting** da molti valori segnalato dalle slide. `marital` (3
  valori) sarebbe più robusto a parità di errori.
- Valutare 1R **sullo stesso file** su cui sceglie l'attributo **non penalizza** l'overfitting:
  il risultato è ottimistico. Solo una valutazione su dati separati (Task 4) lo fa emergere.

1R è il classificatore più semplice e interpretabile: un buon **baseline** da
confrontare con Naïve Bayes.

---

## 3. Classificatore Naïve Bayes

### 3.1 Teoria

Naïve Bayes si basa sulla **regola di Bayes**. Per un'istanza con attributi
$x_1, \dots, x_n$:

$$ P(c \mid x_1,\dots,x_n) \propto P(c)\cdot \prod_{i=1}^{n} P(x_i \mid c) $$

dove $P(c)$ è la probabilità **a priori** della classe e $P(x_i \mid c)$ la
**verosimiglianza** dell'attributo data la classe. Il prodotto assume
l'**indipendenza** degli attributi . Si sceglie la classe col
prodotto più alto (regola **MAP**).

**Trattamento degli attributi nel notebook:**

- **nominali** → frequenze, con **stimatore di Laplace** (conteggi inizializzati a 1):
  $P(x_i \mid c) = \dfrac{\text{conteggio} + 1}{N_c + k}$, dove $k$ è il numero di
  valori distinti dell'attributo. Lo smoothing evita probabilità nulle;
- **numerici** → **discretizzati in fasce** e trattati poi come nominali. Con sole 12
  istanze la stima di una densità gaussiana sarebbe troppo instabile; la discretizzazione
  (`age` in giovane/adulto/senior, `campaign` in basso/medio/alto) rende le stime più robuste.

### 3.2 Selezione delle feature e adattamento ai dati

Con 12 istanze, usare tutti i nominali è controproducente: `education` (7 valori) o `job` (6
valori) avrebbero quasi una sola istanza per valore. Si selezionano quindi **cinque feature**
informative e a bassa cardinalità: `marital`, `housing`, `loan` (nominali) + `age`, `campaign`
(numerici discretizzati).

Il notebook svolge prima il calcolo **a mano su una singola istanza** , mostrando
esplicitamente prior, verosimiglianze con Laplace e combinazione finale. Poi raccoglie tutte le
probabilità condizionate in una struttura dati (scritte esplicitamente come frazioni, per
rispecchiare i calcoli a mano) e definisce `predict_naive_bayes`, che parte dai prior e
**moltiplica un termine alla volta** la verosimiglianza di ogni feature per le due classi,
restituendo la classe MAP.

### 3.3 Risultati e osservazioni

Valutato **sullo stesso `manuale.csv`**, Naïve Bayes ottiene:

| Metrica | Valore |
|---|---|
| Accuracy  | 75.00% |
| Precision | 80.00% |
| Recall    | 66.67% |
| F1-Score  | 72.73% |

Matrice di confusione: TN=5, FP=1, FN=2, TP=4 (9 istanze su 12 corrette).

- Sull'istanza di esempio  Naïve Bayes predice `y=0` mentre la classe reale è `y=1`: un
  **errore**. I punteggi delle due classi erano quasi identici (0.00508 vs 0.00488): con pochi
  dati lo stimatore di Laplace appiattisce le probabilità e basta poco per ribaltare la
  decisione.
- La **precision** (80%) è buona; la **recall** (66.7%) più bassa segnala due falsi negativi —
  l'errore più costoso in marketing, dove si vogliono individuare i clienti interessati.
- Le assunzioni di **indipendenza** e di trattamento omogeneo degli attributi sono forti,
  soprattutto su 12 istanze.

### 3.4 Controprova con Scikit-Learn

Come controprova si usa **`CategoricalNB`** (API scklearn) sulle stesse cinque feature discretizzate e valutato sullo
stesso file. Il risultato (accuratezza 75%) **coincide** con l'implementazione manuale,
confermandone la correttezza. Eventuali minime differenze sono dovute al modo in cui
`CategoricalNB` apprende l'insieme dei valori e gestisce internamente conteggi .

---

## 4. Metodo di valutazione

La valutazione è svolta **sullo stesso file `manuale.csv`** su cui
i modelli sono costruiti. Le metriche utilizzate  sono **Accuracy, Confusion Matrix,
Precision, Recall, F1-Score**, con classe positiva `y = 1`.

> **Avvertenza.** Con sole **12 istanze** e con training = test, le metriche sono **poco
> affidabili** e ottimistiche: qui servono a *illustrare il funzionamento* dei classificatori,
> non a giudicarli. La valutazione seria, su `training.csv` con holdout stratificato e metriche
> adatte allo sbilanciamento, è nei Task 4 e 5.

---

## 5. Confronto e passo successivo

- **1R** (83% su tutte le metriche): semplice e trasparente, ma basato su un solo attributo e
  favorito dall'overfitting da molti valori di `job`. Sul file bilanciato sembra superiore.
- **Naïve Bayes** (F1 72.7%): combina cinque feature e sfrutta più informazione, al costo di
  assunzioni forti; sull'istanza difficile sbaglia per margini minimi.

Su questo piccolo file bilanciato 1R appare migliore, ma proprio perché sfrutta un attributo che
lo favorisce in modo poco robusto. Il confronto **affidabile** è rimandato ai task successivi.

**Prossimo passo (Task 4):** valutare entrambi i classificatori su `training.csv`
(41.176 istanze) con holdout stratificato e metriche adatte allo sbilanciamento, e
ottimizzarne le prestazioni.

---


# Task 2 — Classificatori manuali: 1R e Naïve Bayes

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** `manuale.csv` (12 istanze, generato dal Task 1)
**Codice di riferimento:** `src/task2_1R.ipynb` e `src/task2_NaiveBayes.ipynb`


---

## 1. Obiettivo del task
L'obiettivo è quello di **definire, adattare ai dati,
implementare e valutare a mano due classificatori** su `manuale.csv`. I due scelti
sono **1R** e **Naïve Bayes**, separati in **due notebook distinti** per chiarezza.



**Valutazione sullo stesso file.**
Entrambi i classificatori vengono valutati **sullo stesso `manuale.csv`** su cui sono costruiti.
I risultati ottenuti non sono da considerare rilevanti qualitativamente, in quanto abbiamo lavorato
su un dataset giocattolo: la valutazione statisticamente robusta è nei Task 4 e 5.



**Attributi disponibili in `manuale.csv`:** numerici `age`, `campaign`; nominali `job`, `marital`,
`education`, `housing`, `loan`, `contact`, `poutcome`. Le 12 istanze sono bilanciate 6 `yes` / 6 `no`.

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
notebook si usa un binning semplice sulla **mediana** (due intervalli, `<= mediana` e `> mediana`).

> **Punto critico:** 1R tende all'**overfitting** quando un attributo
> ha **molti valori distinti**. Al limite, un valore diverso per ogni istanza darebbe
> 0 errori ma sarebbe inutile.

### 2.2 Adattamento ai dati e implementazione

- Sono state realizzate due funzioni: `regole_e_errori_attributo`, che implementa 1R per un
  singolo attributo (costruendo la tabella valore × classe con `crosstab` e contando gli errori
  come `totale − maggioranza`), ed `errori_nominale`, che la applica a tutti gli attributi nominali.
- Gli errori si calcolano per **tutti** gli attributi (nominali direttamente,
  numerici dopo binning sulla mediana), iterando con `apply` su una `Series`.
- `addestra_1R` sceglie l'attributo con meno errori e ne memorizza la regola;
  `predici_1R` applica la regola a una nuova istanza (con default `0` se incontra un
  valore mai visto).

Gli errori per attributo risultano: **`marital` 2/12**, `job` 3, `education` 3, `campaign` 4,
`age` 5, `contact` 5, `loan` 5, `housing` 6, `poutcome` 6.

### 2.3 Risultati e osservazioni critiche

L'attributo con meno errori è **`marital`** (2/12, minimo **netto**, senza pareggi). 1R apprende
quindi la regola `{divorced → 1, married → 0, single → 1}`. Valutato **sullo stesso `manuale.csv`**:

| Metrica | Valore |
|---|---|
| Accuracy  | 83.33% |
| Precision | 100.00% |
| Recall    | 66.67% |
| F1-Score  | 80.00% |

Matrice di confusione: TN=6, FP=0, FN=2, TP=4 (10 istanze su 12 corrette).

- 1R classifica correttamente **10 osservazioni su 12** usando **un solo attributo**: per la sua
  semplicità è un risultato notevole, e conferma il ruolo di `marital` come predittore su questo
  piccolo file.
- La **precision è perfetta (100%)**: nessun falso positivo (FP = 0). I due errori sono **falsi
  negativi** (clienti `married` che però hanno sottoscritto), trascinati sulla classe `0` dalla
  regola dominante `married → 0`.
- **Niente overfitting da molti valori:** `marital` ha solo **tre valori distinti**, quindi 1R non
  cade nella tipica trappola dell'attributo ad alta cardinalità (che otterrebbe pochi errori
  "gratis" sul training ma generalizzerebbe male).

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
l'**indipendenza** degli attributi. Si sceglie la classe col
prodotto più alto (regola **MAP**).

**Trattamento degli attributi nel notebook.** Vengono usate **solo feature nominali**, stimate con
**frequenze** e **stimatore di Laplace** (conteggi inizializzati a 1):
$P(x_i \mid c) = \dfrac{\text{conteggio} + 1}{N_c + k}$, dove $k$ è il numero di valori distinti
dell'attributo. Lo smoothing evita probabilità nulle (necessario, perché ad esempio `divorced` e
`single` non compaiono affatto nella classe `0`).

### 3.2 Selezione delle feature e adattamento ai dati

Con sole 12 istanze, usare tutti i nominali è controproducente: `education` (7 valori) o `job` (6
valori) avrebbero quasi una sola istanza per valore. Si selezionano quindi **tre feature** nominali
a bassa cardinalità: **`marital`**, **`housing`** e **`loan`**. I priori, su un file bilanciato,
valgono $P(Y{=}0)=P(Y{=}1)=0.5$.

Il notebook stima le probabilità condizionate per ciascuna delle tre feature (con tabelle
`crosstab` e Laplace), poi le raccoglie in una struttura dati (scritte esplicitamente come
frazioni, per rispecchiare i calcoli a mano) e definisce `predici_naive_bayes`, che parte dai
prior e **moltiplica un termine alla volta** la verosimiglianza di ogni feature per le due classi,
restituendo la classe MAP. Già qui emerge che **`housing` e `loan` sono poco discriminanti**: le
loro distribuzioni sono quasi identiche nelle due classi.

### 3.3 Risultati e osservazioni

Valutato **sullo stesso `manuale.csv`**, Naïve Bayes ottiene:

| Metrica | Valore |
|---|---|
| Accuracy  | 83.33% |
| Precision | 100.00% |
| Recall    | 66.67% |
| F1-Score  | 80.00% |

Matrice di confusione: TN=6, FP=0, FN=2, TP=4 (10 istanze su 12 corrette).

- La **precision perfetta (100%)** indica nessun falso positivo: il modello è molto *prudente* nel
  dichiarare un cliente interessato.
- La **recall (66.7%)** più bassa segnala due falsi negativi — l'errore più costoso in marketing,
  dove si vogliono individuare i clienti propensi. Sono due clienti `married` che hanno sottoscritto:
  il peso della feature dominante `marital` (dove `married` è fortemente associato a `0`) li trascina
  sulla classe sbagliata, mentre `housing` e `loan`, troppo deboli, non riescono a ribaltare la decisione.
- Le assunzioni di **indipendenza** e la base di sole 12 istanze e 3 feature rendono il modello
  fragile: i risultati sono **illustrativi**, non un giudizio affidabile.

### 3.4 Controprova con Scikit-Learn

Come controprova si usa **`CategoricalNB`** (API sklearn) sulle stesse tre feature nominali
(`marital`, `housing`, `loan`), con lo stesso smoothing di Laplace (`alpha=1`), valutato sullo
stesso file. Il risultato (**accuratezza 83.33%**) **coincide** con l'implementazione manuale,
confermandone la correttezza. Eventuali minime differenze sarebbero dovute al modo in cui
`CategoricalNB` apprende l'insieme dei valori e gestisce internamente conteggi e smoothing.

---

## 4. Metodo di valutazione

La valutazione è svolta **sullo stesso file `manuale.csv`** su cui
i modelli sono costruiti. Le metriche utilizzate sono **Accuracy, Confusion Matrix,
Precision, Recall, F1-Score**, con classe positiva `y = 1`.

> **Avvertenza.** Con sole **12 istanze** e con training = test, le metriche sono **poco
> affidabili** e ottimistiche: qui servono a *illustrare il funzionamento* dei classificatori,
> non a giudicarli. La valutazione seria, su `training.csv` con metriche
> adatte allo sbilanciamento, è nei Task 4 e 5.

---

## 5. Confronto e passo successivo

Sullo stesso file bilanciato, **1R e Naïve Bayes producono esattamente le stesse metriche** (Accuracy
83.33%, Precision 100%, Recall 66.67%, F1 80%) e sbagliano le **stesse 2 istanze su 12**:

| Metrica | **1R** (`marital`) | **Naïve Bayes** (3 feature) |
|---|:---:|:---:|
| Accuracy  | 83.33% | 83.33% |
| Precision | 100.00% | 100.00% |
| Recall    | 66.67% | 66.67% |
| F1-Score  | 80.00% | 80.00% |
| Errori    | 2 / 12 | 2 / 12 |

**Non è una coincidenza.** Nel Naïve Bayes i priori sono uguali (0.5 / 0.5) e si elidono;
`housing` ha distribuzioni identiche nelle due classi e si semplifica; `loan` è troppo debole per
ribaltare la decisione. Di fatto la predizione è guidata dal **solo `marital`**, esattamente
l'attributo su cui 1R costruisce la propria regola: ecco perché i due modelli prendono le **stesse**
decisioni. Un Naïve Bayes costruito su poche feature poco discriminanti **degenera** in un
classificatore a regola singola, indistinguibile da 1R.

**Prossimo passo (Task 4):** valutare entrambi i classificatori su `training.csv`
(41.176 istanze) con metriche adatte allo sbilanciamento, e
discuterne i margini di ottimizzazione. La differenza tra i due approcci emergerà davvero nel
**Task 5**, dove Naïve Bayes userà tutte le feature, verrà riaddestrato e avrà un preprocessing completo.

---

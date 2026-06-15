# Task 2 — Classificatori manuali: 1R e Naïve Bayes

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** `manuale.csv` (12 istanze, generato dal Task 1)
**Codice di riferimento:** `src/task2_1R.ipynb` e `src/task2_NaiveBayes.ipynb`
**Libro di testo:** Witten, Frank, Hall, Pal — *Data Mining* (4ª ed.), cap. 4 — Lezione 5

---

## 1. Obiettivo del task

La traccia, per un gruppo da due persone, chiede di **definire, adattare ai dati,
implementare e valutare a mano due classificatori** su `manuale.csv`. I due scelti
sono **1R** e **Naïve Bayes**, entrambi visti nella Lezione 5. I due classificatori
sono stati separati in **due notebook distinti** per chiarezza.

**Stile di codice.** Vettoriale (pandas/numpy: `groupby`, `crosstab`, `value_counts`,
`apply`, `prod`, `idxmax`), **senza cicli `for` espliciti**, coerentemente con i
notebook del corso.

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

> **Punto critico (dalle slide):** 1R tende all'**overfitting** quando un attributo
> ha **molti valori distinti**. Al limite, un valore diverso per ogni istanza darebbe
> 0 errori ma sarebbe inutile.

### 2.2 Adattamento ai dati e implementazione

- `regola_1R(serie_attr, y)` costruisce la regola (`groupby` + `mode`) e conta gli
  errori con `crosstab`: per ogni valore, errori = totale − conteggio della classe a
  maggioranza.
- Gli errori si calcolano per **tutti** gli attributi (nominali direttamente,
  numerici dopo binning sulla mediana), iterando con `apply` su una `Series`.
- `addestra_1R` sceglie l'attributo con meno errori e ne memorizza la regola;
  `predici_1R` applica la regola a una nuova istanza (con default `0` se incontra un
  valore mai visto).

### 2.3 Risultati e osservazioni critiche

- 1R sceglie l'attributo **`job`** (2/12 errori sul training, ~83% di accuratezza).
- **Pareggio tra `job` e `marital`** (entrambi 2 errori): 1R sceglie il primo per
  *tie-breaking*. Tuttavia `job` ha **4 valori con una sola istanza**, ognuno dei
  quali dà 0 errori "gratis": è l'**overfitting** da molti valori segnalato dalle
  slide. `marital` (3 valori) sarebbe più robusto a parità di errori.
- In **leave-one-out** (vedi §4) l'accuratezza scende sensibilmente: il divario tra
  training e leave-one-out è la prova concreta dell'overfitting.

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
l'**indipendenza** degli attributi (da qui "naïve"). Si sceglie la classe col
prodotto più alto (regola **MAP**).

**Due tipi di attributo:**

- **nominali** → frequenze, con **stimatore di Laplace** (conteggi inizializzati a 1):
  $P(x_i \mid c) = \dfrac{\text{conteggio} + 1}{N_c + v_i}$, dove $v_i$ è il numero di
  valori distinti dell'attributo. Lo smoothing evita probabilità nulle;
- **numerici** → **distribuzione gaussiana**:
  $P(x_i \mid c) = \dfrac{1}{\sqrt{2\pi}\,\sigma}\, e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}$,
  con $\mu$ e $\sigma$ stimati per classe.

### 3.2 Adattamento ai dati e implementazione

Il notebook svolge prima il calcolo **a mano su una singola istanza** (la riga 0,
addestrando sulle altre 11), mostrando esplicitamente prior, verosimiglianze
nominali (Laplace) e numeriche (gaussiana), e la combinazione finale. Poi
generalizza in `naive_bayes_score`, completamente vettoriale: `crosstab` per i
nominali, `groupby` per le statistiche dei numerici, `.prod()` per il prodotto delle
verosimiglianze, `idxmax` per la classe MAP.

### 3.3 Osservazioni

- Sull'istanza di esempio (riga 0) Naïve Bayes predice `y=0` mentre la classe reale è
  `y=1`: un **errore**. Con così pochi dati le stime di probabilità sono instabili e
  alcuni attributi nominali "tirano" verso la classe 0.
- Le assunzioni di **indipendenza** degli attributi e di **normalità** dei numerici
  sono forti, soprattutto su 12 istanze.
- Rispetto a 1R, Naïve Bayes **sfrutta tutti gli attributi**: ci si attende che
  catturi più informazione.

### 3.4 Controprova con Scikit-Learn

La traccia consente l'uso di API. Come controprova si usa `GaussianNB` con
`OrdinalEncoder` sui nominali, valutato in leave-one-out tramite `cross_val_predict`
+ `LeaveOneOut` (così si evitano cicli espliciti). Il risultato non è identico
all'implementazione manuale, perché sklearn tratta i nominali come numeri e applica a
tutti la gaussiana, mentre la versione a mano distingue nominali (Laplace) e numerici
(gaussiana).

---

## 4. Metodo di valutazione

Con sole **12 istanze** si usa il **leave-one-out** (addestrare su 11, testare su 1,
ripetuto per ogni istanza): è il modo corretto di stimare le prestazioni su pochi
dati ed è il caso estremo della cross-validation (Lezione 8). L'iterazione è fatta
con `apply` sull'indice, senza `for`.

> **Avvertenza.** Le metriche su 12 istanze sono **poco affidabili**: qui servono a
> *illustrare il funzionamento* dei classificatori, non a giudicarli. La valutazione
> seria, su `training.csv`, è nei Task 4 e 5.

---

## 5. Confronto e passo successivo

- **1R**: semplice, trasparente, ma basato su un solo attributo; soffre overfitting
  su attributi con molti valori e cattura male la classe rara.
- **Naïve Bayes**: combina tutti gli attributi e tende a sfruttare più informazione,
  al costo di assunzioni forti.

**Prossimo passo (Task 4):** valutare entrambi i classificatori su `training.csv`
(41.176 istanze) con holdout stratificato e metriche adatte allo sbilanciamento, e
ottimizzarne le prestazioni.

---

## 6. Concetti del corso utilizzati

| Concetto | Lezione |
|---|---|
| Attributi numerici vs nominali | 3 |
| 1R, discretizzazione, overfitting da molti valori | 5 |
| Naïve Bayes, regola di Bayes, stimatore di Laplace, gaussiana | 5 |
| Leave-one-out / cross-validation | 8 |
| Uso di API (GaussianNB) come controprova | trasversale |

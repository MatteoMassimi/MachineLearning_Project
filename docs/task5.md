# Task 5 — Addestramento di più classificatori con Scikit-Learn

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** Bank Marketing (UCI) — previsione della sottoscrizione di un deposito vincolato
**Codice di riferimento:** `src/task5.ipynb`

---

## 1. Obiettivo del task

La traccia richiede, con riferimento a `training.csv`, di:

1. addestrare tramite Scikit-Learn **più classificatori**;
2. separare opportunamente i campioni in **training set** e **test set**;
3. **massimizzare le prestazioni sul test set**;
4. **descrivere e analizzare con spirito critico** le decisioni e i risultati;
5. **selezionare il classificatore più performante**, che in sede d'esame sarà testato
   sul file `real_settings.csv`.

I **cinque classificatori** considerati sono: **Naïve Bayes**, **Regressione logistica**,
**Albero di decisione**, **k-NN** e **Percettrone**. Il codice completo, eseguibile e commentato,
si trova nel notebook `src/task5.ipynb`.

---

## 2. Feature escluse: `duration` e `pdays`

Due esclusioni, per ragioni diverse, entrambe già motivate nei task precedenti:

- **`duration`** è *data leakage* (la durata della chiamata è nota solo *dopo* la chiamata) ed è
  stata **rimossa già nel Task 1**. Non compare quindi in `training.csv` e non richiede alcun
  intervento qui.
- **`pdays`** vale 999 ("mai contattato") per il ~96% delle istanze (Task 3): è una
  **near-constant feature**, poco informativa, e viene **esclusa dalle feature** in questo task
  (`X = training_df.drop(columns=["y", "pdays"])`). Restano **18 feature** in ingresso.

---

## 3. Separazione train/test: holdout 70/30 stratificato

Seguiamo lo schema della **Lezione 8** (*Valutazione*):

- **Holdout 70/30**: il 70% dei dati per addestrare (28.823 campioni), il 30% — mai visto in
  addestramento — per la valutazione finale e imparziale (12.353 campioni).
- **`stratify=y`**: mantiene la **stessa proporzione di classi** (~11% "yes", esattamente 0.1127)
  sia nel training sia nel test. È essenziale con classi sbilanciate: senza stratificazione il
  test potrebbe contenere troppi pochi "yes" e dare stime instabili.
- **`random_state=10`**: rende lo split riproducibile, in linea con la convenzione dei
  notebook del corso (`SEED = 10`).

Il test set di holdout resta **completamente da parte**: è il sostituto "locale" del
futuro `real_settings.csv`. La cross-validation (Sezione 7) e il tuning (Sezione 8) girano
*dentro* il solo training set, senza mai toccare il test.

---

## 4. Preprocessing tramite `ColumnTransformer`

Il cuore tecnico del task è un **`ColumnTransformer`** che applica il preprocessing corretto a
ciascun tipo di attributo (Lezione 3 / 11):

| Ramo | Trasformazione | Perché |
|---|---|---|
| **Numerico** (8 colonne) | `StandardScaler` | k-NN (distanze), Regressione logistica e Percettrone sono sensibili alla scala |
| **Nominale** (10 colonne) | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` | i classificatori lavorano su numeri, non su stringhe |

Dopo l'encoding la matrice ha **55 colonne** (8 numeriche + 47 dummy one-hot).

**Niente imputazione qui:** i valori `unknown` sono già stati imputati con la moda nel Task 1,
quindi il preprocessing si limita a scaling e one-hot.

**Due modalità d'uso nel notebook:**

- nelle **Sezioni 5–6** il preprocessor viene **adattato una sola volta** sul training (`fit_transform`)
  e applicato al test (`transform`), producendo le matrici `X_addestramento_enc` / `X_test_enc`
  passate direttamente ai classificatori. Adattarlo solo sul training evita il *data leakage*;
- dalla **Sezione 7** in poi lo stesso preprocessor viene inserito in una **`Pipeline`** col
  classificatore, così da essere ri-adattato **dentro ogni fold** della cross-validation — l'unico
  modo corretto di validare e fare tuning senza leakage.

`handle_unknown="ignore"` evita errori se nel test (o in `real_settings.csv`) comparisse una
categoria mai vista in addestramento.

---

## 5. Gestione dello sbilanciamento e metrica guida

Il dataset ha circa l'**11% di "yes"**. Da qui due conseguenze:

- **L'accuracy è fuorviante:** un classificatore banale "sempre no" raggiunge ~88.7% di accuracy
  senza imparare nulla. La metrica guida è quindi la **F1 sulla classe positiva "yes"** (media
  armonica di precision e recall), affiancata da precision, recall, accuratezza e matrice di confusione.
- **Si ribilancia il costo degli errori:** si adotta **`class_weight="balanced"`** nei modelli che
  lo supportano — **Regressione logistica, Albero di decisione, Percettrone** — penalizzando di più
  gli errori sulla classe rara, senza inventare dati. **Naïve Bayes** e **k-NN** *non* dispongono di
  `class_weight`: su di essi lo sbilanciamento non viene corretto in addestramento, e ne risentono
  (k-NN in particolare tende a predire "no").

La scelta della F1 nasce dalla **logica di business**: il falso negativo (cliente propenso non
chiamato → ricavo perso) pesa più del falso positivo (chiamata a vuoto), ma chiamare *tutti* è
insostenibile. Serve un equilibrio tra precision e recall → si ottimizza la **F1**.

---

## 6. Confronto dei cinque classificatori (holdout)

Risultati sul test set, ordinati per F1 sulla classe "yes":

| Modello | Accuracy | Precision | Recall | **F1** |
|---|---|---|---|---|
| **Albero di decisione** (`max_depth=5`, balanced) | 0.870 | 0.439 | 0.561 | **0.492** |
| **Regressione logistica** (balanced) | 0.831 | 0.358 | 0.634 | 0.458 |
| k-NN (`k=31`) | 0.896 | 0.602 | 0.234 | 0.336 |
| Naïve Bayes (`GaussianNB`) | 0.624 | 0.198 | 0.764 | 0.314 |
| Percettrone (balanced) | 0.847 | 0.307 | 0.284 | 0.295 |

**Lettura lungo l'asse precision/recall:**

- **Naïve Bayes** e **k-NN** sono i due **estremi**: il primo ha la recall più alta (0.764) ma
  precision minima (0.198) — tanti falsi allarmi, accuratezza sotto il baseline; il secondo è lo
  specchio opposto (precision 0.602, recall 0.234) — prudente ma si perde i tre quarti dei positivi,
  perché senza `class_weight` il vicinato è dominato dalla classe maggioritaria.
- **Albero, Regressione logistica** e (più debolmente) **Percettrone**, grazie a
  `class_weight="balanced"`, recuperano recall mantenendo una precision sostenibile: è il
  **trade-off desiderato** per una campagna di marketing.
- **L'accuracy conferma di ingannare:** il k-NN ha l'accuratezza più alta (0.896, unico sopra il
  baseline) ma F1 modesta; i due modelli con F1 migliore (Albero, Logistica) hanno accuratezza
  *inferiore* al baseline. Su dati sbilanciati l'accuratezza misura il conformismo, non la capacità
  di individuare la classe rara.

`GaussianNB` è applicato sull'intera matrice one-hot per uniformità di pipeline: modella ogni
feature come gaussiana e assume indipendenza, ipotesi entrambe violate dalle dummy binarie e dalle
variabili economiche correlate — da cui il profilo "recall alta / precision bassa".

---

## 7. Cross-validation a 10 fold

La valutazione holdout dipende dalla particolare suddivisione. Per una stima più robusta si applica
una **Stratified 10-Fold Cross Validation** (Lezione 9) **solo sul training set**, con `Pipeline`
(preprocessing + modello) ri-adattata in ogni fold e scoring **F1**. I cinque modelli sono raccolti
in un dizionario e valutati in modo **vettoriale** con `Series.apply` (vincolo di stile: nessun
ciclo `for` esplicito).

| Modello | F1 media (CV) | Dev. std |
|---|---|---|
| Albero di decisione | 0.473 | ±0.031 |
| Regressione logistica | 0.453 | **±0.015** |
| k-NN | 0.332 | ±0.031 |
| Naïve Bayes | 0.319 | ±0.005 |
| Percettrone | 0.293 | **±0.067** |

**L'ordine regge:** la classifica è identica a quella dell'holdout — gli esiti non erano un
artefatto della particolare suddivisione. La CV aggiunge la dimensione della **stabilità**:

- l'**Albero** ha la media più alta, ma la **Logistica** è in pratica appaiata (Δ ≈ 0.020) ed è
  **circa il doppio più stabile** (±0.015 vs ±0.031): per questo entrambi vengono portati al tuning;
- il **Percettrone** ha una dispersione enorme (±0.067, oltre 4× la logistica): conferma la sua
  instabilità di modello lineare "duro" che non converge su dati non linearmente separabili.

**Assenza di overfitting:** per i modelli di testa la F1 in CV è molto vicina a quella sul test (es.
albero 0.473 vs 0.492), segno che non c'è memorizzazione del training.

> Rispetto a una 5-fold, la 10-fold addestra su una frazione maggiore dei dati (~90%), riducendo il
> *bias* della stima; in cambio ogni fold di validazione è più piccolo (F1 leggermente più "rumorosa").

---

## 8. Ottimizzazione degli iperparametri (`GridSearchCV`)

Il tuning usa `GridSearchCV` (stessa CV stratificata a 10 fold, scoring **F1**) sui **tre modelli**
con maggiore potenziale o più sensibili agli iperparametri. **Naïve Bayes** e **Percettrone** sono
**esclusi** (il primo non ha iperparametri che cambino il trade-off su questa matrice one-hot, il
secondo è già il più debole e instabile). Anche qui niente cicli espliciti: le ricerche sono in un
dizionario `{nome: (pipeline, griglia)}` lanciato con `Series.apply`.

| Modello | Griglia | Selezionato | F1 CV | F1 test |
|---|---|---|---|---|
| Regressione logistica | `C ∈ {0.01, 0.1, 1, 10, 100}` | `C = 100` | 0.454 | 0.458 |
| Albero di decisione | `max_depth ∈ {3, 5, 7, 10, None}` | `max_depth = 5` | 0.473 | 0.492 |
| k-NN | `n_neighbors ∈ {15, 31, 51}` | `k = 15` | 0.350 | 0.353 |

**Messaggio trasversale del tuning:**

- **Albero** — la profondità scelta a mano (`max_depth=5`) si **riconferma** ottimale: alberi più
  profondi non generalizzano, `max_depth=3` è troppo rigido. Nessun guadagno residuo.
- **Regressione logistica** — `C` è **quasi irrilevante**: cinque ordini di grandezza muovono la F1
  di pochi millesimi. Il limite non è la regolarizzazione ma l'**informazione disponibile nelle
  feature** (senza `duration`).
- **k-NN** — un vicinato più stretto (`k=15`) dà il guadagno maggiore (F1 0.336 → 0.353), ma il
  modello resta strutturalmente sbilanciato verso la precision.

I guadagni rispetto alle configurazioni iniziali sono **marginali o nulli**: le scelte "a mano"
erano già sensate, e la piattezza delle griglie indica che il collo di bottiglia è il **contenuto
informativo dei dati**, non l'ottimizzazione.

---

## 9. Selezione del modello finale

Il modello finale è quello con la **F1 più alta sul test** (selezione automatica via `idxmax`): è
l'**Albero di decisione** ottimizzato (`max_depth=5`, F1 test **0.492**). Il notebook ne analizza
**matrice di confusione** e **classification report**, per capire *come* sbaglia e non solo *quanto*.

---

## 10. Curva di apprendimento

Sul modello selezionato si traccia una **curva di apprendimento** (Lezione 10): la F1 su training e
in cross-validation al crescere del numero di campioni. Il risultato:

- le due curve sono **molto vicine** (gap finale di pochi millesimi) e si **appiattiscono presto**
  (già intorno agli 8.000 campioni) su un valore di F1 contenuto (~0.49);
- è il profilo tipico dell'**underfitting (alto bias)**, non dell'overfitting: il modello non sta
  memorizzando il training, ma è **limitato dall'informazione nelle feature**.

**Conclusione operativa:** raccogliere più campioni **non aiuterebbe** (il tetto è già raggiunto); il
margine va cercato altrove — feature più informative (la rimozione di `duration` ha tolto il
predittore più forte), feature engineering, o lo spostamento della soglia decisionale per
privilegiare la recall. Il limite di ~0.49 di F1 è una caratteristica **del dataset** in condizioni
realistiche (senza leakage), non un difetto del classificatore: un risultato onesto e ben diagnosticato.

---

## 11. Predisposizione per `real_settings.csv`

Il modello selezionato è una **pipeline completa già addestrata** (`preprocessing → classificatore`).
Ricevuto `real_settings.csv`, **non** serve rifare manualmente encoding e scaling: basta caricarlo,
rimuovere `duration`/`pdays` se presenti, e chiamare `predict`. La pipeline applica automaticamente
le **stesse** trasformazioni apprese sul training, gestendo anche categorie nuove
(`handle_unknown="ignore"`). Se il file conterrà la colonna `y`, si valuteranno le prestazioni con
le stesse metriche usate qui. Il notebook contiene la cella-template pronta da eseguire.

---

## 12. Domande d'esame probabili

**D — Perché hai escluso `duration`?**
È *data leakage*: la durata si conosce solo dopo la chiamata. È stata rimossa già nel Task 1, così
tutti i task lavorano su dati realistici; usarla gonfierebbe i punteggi in laboratorio ma il modello
crollerebbe su `real_settings.csv`.

**D — Perché holdout *e* cross-validation insieme?**
Ruoli diversi: la CV serve per il *tuning* degli iperparametri e gira solo sul training; l'holdout
fornisce la stima *finale e imparziale* su dati mai visti. Fare tuning sul test sarebbe leakage.

**D — Perché la pipeline e non trasformare i dati una volta sola all'inizio?**
Per evitare leakage nel preprocessing: scaling e encoding devono apprendere i parametri solo dal
training. La pipeline garantisce che ciò avvenga correttamente anche dentro ogni fold della CV.

**D — Perché `class_weight="balanced"`?**
Ripesa la *loss* a favore della classe rara senza alterare i dati. Si applica a Logistica, Albero e
Percettrone; Naïve Bayes e k-NN non lo prevedono e per loro si potrebbe agire sulla soglia decisionale.

**D — Perché ottimizzi la F1 e non l'accuracy?**
Con l'11% di "yes" l'accuracy premia il classificatore banale "sempre no" (~88.7%). La F1 sulla
classe positiva misura l'equilibrio tra precision e recall sulla classe che interessa davvero.

**D — Perché 10 fold e non 5?**
Addestra ogni modello su ~90% dei dati (meno *bias* nella stima della media); con ~29.000 campioni il
costo doppio resta sostenibile. Le medie sono comunque quasi identiche a una 5-fold: stima robusta.

**D — La curva di apprendimento cosa dice?**
Underfitting (alto bias): training e validazione vicine e basse, appiattite presto. Più dati non
aiutano; il limite è informativo, intrinseco al dataset senza `duration`.

---

## 13. Concetti del corso utilizzati

| Concetto | Lezione |
|---|---|
| Attributi numerici vs nominali | 3 |
| Naïve Bayes, k-NN, alberi, modelli lineari (logistica, percettrone) | 5–7 |
| Holdout, stratificazione, metriche, confusion matrix | 8 |
| Cross-validation, tuning (`GridSearchCV`) | 9 |
| Curva di apprendimento (bias/varianza) | 10 |
| One-hot encoding, scaling | 11 |
| Data leakage | trasversale |
| Sbilanciamento delle classi e `class_weight` | trasversale |

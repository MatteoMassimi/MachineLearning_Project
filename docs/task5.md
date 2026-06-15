# Task 5 — Addestramento di più classificatori con Scikit-Learn

**Corso:** Fondamenti e Applicazioni del Machine Learning (FML 2026)
**Dataset:** Bank Marketing (UCI) — previsione della sottoscrizione di un deposito vincolato
**Codice di riferimento:** `src/task5.ipynb`
**Libro di testo:** Witten, Frank, Hall, Pal — *Data Mining* (4ª ed.)

---

## 1. Obiettivo del task

La traccia richiede, con riferimento a `training.csv`, di:

1. addestrare tramite Scikit-Learn **più classificatori**;
2. separare opportunamente i campioni in **training set** e **test set**;
3. **massimizzare le prestazioni sul test set**;
4. **descrivere e analizzare con spirito critico** le decisioni e i risultati, usando
   il maggior numero di concetti visti a lezione;
5. **selezionare il classificatore più performante**, che in sede d'esame sarà testato
   sul file `real_settings.csv`.

Questo documento descrive le decisioni prese e le loro motivazioni. Il codice completo,
eseguibile e commentato, si trova nel notebook `src/task5.ipynb`.

---

## 2. La decisione più importante: esclusione di `duration` (data leakage)

È la scelta metodologica centrale del task e il primo punto da saper difendere all'orale.

### Il problema

La variabile `duration` (durata in secondi dell'ultima telefonata) era, nelle analisi
dei Task 3 e 4, la feature **più correlata con il target**. Questa correlazione è però
**ingannevole**:

- la durata della chiamata si conosce **solo dopo** averla effettuata;
- una chiamata lunga è *conseguenza* dell'interesse del cliente, non una sua *causa*
  predittiva: chi è propenso a sottoscrivere resta più a lungo al telefono;
- usarla per *predire* la sottoscrizione equivale a usare informazione proveniente dal
  futuro. Questo è **data leakage**: fuga di informazione dal target verso le feature.

### Il legame con `real_settings.csv`

Il file di test d'esame simula **condizioni reali**: prima di telefonare a un cliente,
`duration` vale 0 o è ignota. Un modello che vi si appoggia otterrebbe ottimi punteggi in
laboratorio ma **crollerebbe** sul test reale. Per costruire un classificatore davvero
utile, lo escludiamo dal modeling.

### Coerenza con i task precedenti

L'avevamo deliberatamente **incluso** nell'EDA (Task 3) e nella valutazione manuale
(Task 4) *proprio per dimostrarne* l'effetto. Qui, nel passaggio alla modellazione finale,
lo rimuoviamo: è il punto corretto del flusso di progetto in cui farlo.

---

## 3. Separazione train/test: holdout 70/30 stratificato

Seguiamo lo schema della **Lezione 8** (*Valutazione*):

- **Holdout 70/30**: il 70% dei dati per addestrare, il 30% — mai visto in addestramento
  — per la valutazione finale e imparziale.
- **`stratify=y`**: mantiene la **stessa proporzione di classi** (~11% "yes") sia nel
  training sia nel test. È essenziale con classi sbilanciate: senza stratificazione il
  test potrebbe contenere troppi pochi "yes" e dare stime instabili.
- **`random_state=10`**: rende lo split riproducibile, in linea con la convenzione dei
  notebook del corso (`np.random.seed(10)`).

Il test set di holdout resta **completamente da parte**: è il sostituto "locale" del
futuro `real_settings.csv`. La cross-validation (Sezione 6) verrà usata *dentro* il solo
training set, per il tuning, senza mai toccare il test.

---

## 4. Preprocessing tramite pipeline (`ColumnTransformer`)

Il cuore tecnico del task è una **pipeline** che applica il preprocessing corretto a
ciascun tipo di attributo, distinguendo (Lezione 3) tra numerici e categorici.

### Perché una pipeline e non trasformazioni manuali

È il modo corretto per **evitare data leakage durante il preprocessing**. Se calcolassimo
media e deviazione per lo scaling sull'intero dataset *prima* dello split, le statistiche
del test contaminerebbero il training. La pipeline invece:

- apprende i parametri di trasformazione (medie, categorie) **solo sul training**, in `fit`;
- li **applica** identici al test, in `transform`;
- durante la cross-validation del tuning, **ricalcola** le trasformazioni a ogni fold in
  modo corretto.

### I due rami del `ColumnTransformer`

| Ramo | Step | Riferimento |
|---|---|---|
| **Numerico** | `SimpleImputer(median)` → `StandardScaler` | Lezione 11 |
| **Categorico** | `SimpleImputer(most_frequent)` → `OneHotEncoder` | Lezione 11 |

Lo **scaling** dei numerici è necessario perché KNN (distanze) e Regressione logistica
(regolarizzazione) sono sensibili alla scala. L'**one-hot encoding** serve perché i
classificatori lavorano su numeri, non su stringhe.

### Dettaglio tecnico: `sparse_output=False`

L'one-hot encoding produce per default una matrice *sparsa*, ma `GaussianNB` richiede dati
*densi*. Impostiamo quindi `OneHotEncoder(sparse_output=False)`: l'overhead di memoria è
trascurabile (poche decine di colonne) e così tutti e cinque i modelli condividono la
stessa pipeline. `handle_unknown="ignore"` evita errori se nel test comparisse una
categoria mai vista in addestramento — utile in vista di `real_settings.csv`.

---

## 5. Gestione dello sbilanciamento delle classi

Il dataset ha circa l'**11% di "yes"**. Da qui due conseguenze, già emerse nei task
precedenti.

### Conseguenza 1 — l'accuracy è fuorviante

Un classificatore banale che predice **sempre "no"** raggiunge ~89% di accuracy senza
imparare nulla. L'accuracy da sola è quindi inutile: guardiamo **precision, recall e F1
sulla classe minoritaria "yes"**, quella di interesse (i clienti da contattare).

### Conseguenza 2 — ribilanciare il costo degli errori

Adottiamo **`class_weight="balanced"`** nei modelli che lo supportano (Albero, Regressione
logistica, Random Forest). Il parametro ripesa la *funzione di costo* penalizzando di più
gli errori sulla classe rara, **senza inventare dati**.

### Perché `class_weight` e non SMOTE (oversampling)

- **Pulizia metodologica**: agisce dentro l'ottimizzazione del modello, non altera il
  dataset.
- **Coerenza con i dati**: SMOTE *sintetizza* campioni interpolando i vicini; con molte
  feature categoriche one-hot, le interpolazioni producono combinazioni irrealistiche
  (es. frazioni di "sposato" e "divorziato" insieme).
- **Aderenza al corso**: SMOTE richiede la libreria esterna `imbalanced-learn`, non vista
  a lezione; la traccia premia l'uso dei *metodi del corso*.

### Limite dichiarato

Due modelli — **KNN** e **Naïve Bayes** — *non* dispongono di `class_weight`. Su di essi lo
sbilanciamento non viene corretto in addestramento, e ne risentono (KNN in particolare
tende a predire "no"). Una leva alternativa per questi modelli è lo spostamento della
**soglia decisionale** (Sezione 8). Dichiarare questo limite è un punto di consapevolezza
critica, non un difetto da nascondere.

---

## 6. I cinque classificatori e l'ottimizzazione degli iperparametri

Confrontiamo **cinque** modelli, tutti visti a lezione (Lezioni 5–7), ciascuno con una
**griglia di iperparametri** contenuta ma significativa, esplorata via `GridSearchCV` con
**cross-validation stratificata a 5 fold** (Lezione 9). Il tuning è la risposta diretta
alla richiesta della traccia di *"massimizzare le prestazioni"*.

| Modello | Iperparametri esplorati | `class_weight`? |
|---|---|---|
| Naïve Bayes (Gaussiano) | `var_smoothing` | no (non disponibile) |
| KNN | `n_neighbors`, `weights` | no (non disponibile) |
| Albero di decisione | `max_depth`, `min_samples_leaf` | sì |
| Regressione logistica | `C` (regolarizzazione) | sì |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` | sì |

I modelli sono organizzati in un dizionario `{nome: (stimatore, griglia)}`, su cui si itera
in modo **vettorizzato** (`Series.apply` su un `pd.Index`), rispettando il vincolo di stile
del progetto: **nessun ciclo `for` esplicito, nessuna list comprehension**.

---

## 7. Scelta della metrica di ottimizzazione

La scelta nasce dalla **logica di business** del problema. Il modello serve a decidere
**quali clienti chiamare**, e i due errori non sono simmetrici:

- **Falso negativo** (cliente propenso non chiamato) → **ricavo perso**;
- **Falso positivo** (cliente non interessato chiamato) → **costo di una chiamata sprecata**.

I falsi negativi pesano di più, ma chiamare *tutti* è insostenibile. Serve un **equilibrio
tra precision e recall** sulla classe "yes" → ottimizziamo la **F1 della classe positiva**
(`scoring="f1"`), media armonica delle due e metrica naturale per la minoranza su dati
sbilanciati.

Come metriche **diagnostiche** di supporto riportiamo:

- **PR-AUC** (*average precision*): a differenza della ROC-AUC è sensibile allo
  sbilanciamento e descrive meglio la qualità del *ranking* sulla classe rara;
- **ROC-AUC** e **accuracy**: per completezza e per mostrare *perché* l'accuracy inganna.

---

## 8. Selezione del modello finale e ruolo della soglia decisionale

Il modello finale è quello con la **F1 più alta sul test set** (selezione via `idxmax`).
Di esso il notebook analizza in dettaglio **matrice di confusione** e **classification
report**, per capire *come* sbaglia, non solo *quanto*.

La **curva Precision-Recall** mostra il compromesso tra le due metriche al variare della
**soglia** sulla probabilità predetta. Questo collega il punto lasciato aperto nella
Sezione 5: per i modelli senza `class_weight` (KNN, Naïve Bayes), o quando si vuole
privilegiare la *recall* (non perdere clienti propensi), si può **abbassare la soglia**
sotto il classico 0.5. La curva è lo strumento per scegliere quella soglia in modo
consapevole, in base agli obiettivi di campagna.

---

## 9. Risultati e analisi critica

> **Nota.** I valori numerici precisi si ottengono eseguendo il notebook su `training.csv`.
> La lettura qualitativa che segue resta valida indipendentemente dai numeri esatti. La
> tabella va completata con i risultati della propria esecuzione.

| Modello | F1 (CV) | F1 (test) | PR-AUC | ROC-AUC | Accuracy |
|---|---|---|---|---|---|
| Regressione logistica | … | … | … | … | … |
| Random Forest | … | … | … | … | … |
| Naïve Bayes | … | … | … | … | … |
| Albero di decisione | … | … | … | … | … |
| KNN | … | … | … | … | … |

Osservazioni critiche principali:

**1. L'accuracy conferma di essere fuorviante.** Tutti i modelli superano o sfiorano il
baseline "always no" (~89%) in accuratezza, eppure si comportano in modo molto diverso
sulla classe utile. È la dimostrazione pratica del perché, su dati sbilanciati, la metrica
giusta è la F1 sulla minoranza.

**2. Il ribilanciamento via `class_weight` funziona.** I modelli che lo supportano
ottengono una **recall sulla classe "yes" sostanzialmente più alta** rispetto a un modello
non bilanciato, al prezzo di una precision più bassa. È il trade-off *desiderato*: in una
campagna di marketing si preferiscono alcune chiamate in più a basso costo piuttosto che
perdere clienti propensi.

**3. KNN è il punto debole, come previsto.** Senza `class_weight`, tende a classificare
quasi tutto come "no" (la classe maggioritaria domina il vicinato), con F1 sulla minoranza
molto bassa. Il motivo era stato anticipato nella Sezione 5. Sarebbe recuperabile
abbassando la soglia decisionale o pesando i vicini per distanza, ma resta strutturalmente
penalizzato dallo sbilanciamento.

**4. Modelli lineari ed ensemble dominano.** Tipicamente Regressione logistica e Random
Forest emergono come i migliori: la prima perché `class_weight` agisce direttamente sulla
*loss* regolarizzata; la seconda per la capacità di catturare interazioni non lineari (es.
`poutcome` × indicatori economici).

**5. Multicollinearità.** Gli indicatori economici (`emp.var.rate`, `euribor3m`,
`nr.employed`) sono fortemente correlati tra loro (>0.90, osservato nel Task 3). Questo
**non danneggia** Random Forest, ma rende **instabili i coefficienti** della Regressione
logistica, da interpretare con cautela (la regolarizzazione `C` mitiga il problema).

**6. Coerenza con i task manuali.** I risultati confermano la direzione dei Task 2 e 4:
Naïve Bayes scambia accuracy per migliore recall sulla minoranza, mentre i metodi basati su
regole/alberi faticano sullo sbilanciamento se non corretti.

---

## 10. Predisposizione per `real_settings.csv`

Il modello selezionato è una **pipeline completa** (`preprocessing → classificatore`).
Questo è un vantaggio decisivo per l'esame: ricevuto `real_settings.csv`, **non** serve
rifare manualmente encoding e scaling. Basterà caricarlo, rimuovere `duration` se presente,
e chiamare `predict`. La pipeline applica automaticamente le **stesse** trasformazioni
apprese sul training, gestendo anche categorie nuove (`handle_unknown="ignore"`).

Il notebook contiene la cella-template pronta da eseguire in sede d'esame.

---

## 11. Domande d'esame probabili

**D — Perché hai escluso `duration` solo qui e non nei task precedenti?**
Perché nei Task 3 e 4 serviva a *dimostrare* il fenomeno della data leakage; nel Task 5,
che produce il modello da usare in condizioni reali (`real_settings.csv`), va rimossa per
non costruire un classificatore inutilizzabile in produzione.

**D — Perché holdout *e* cross-validation insieme?**
Hanno ruoli diversi: la cross-validation serve per il *tuning* degli iperparametri e gira
solo sul training set; l'holdout fornisce la stima *finale e imparziale*, su dati mai visti.
Mescolarli (fare tuning sul test) sarebbe leakage.

**D — Perché la pipeline e non trasformare i dati una volta sola all'inizio?**
Per evitare leakage nel preprocessing: scaling e encoding devono apprendere i parametri
solo dal training. La pipeline garantisce che ciò avvenga correttamente anche dentro ogni
fold della cross-validation.

**D — Perché `class_weight` e non SMOTE?**
`class_weight` è più pulito (non altera i dati), più coerente con feature categoriche
(SMOTE interpola creando combinazioni irrealistiche) e usa metodi del corso (SMOTE richiede
una libreria esterna).

**D — Perché ottimizzi la F1 e non l'accuracy?**
Perché con l'11% di "yes" l'accuracy premia il classificatore banale "always no" (~89%). La
F1 sulla classe positiva misura l'equilibrio tra precision e recall sulla classe che
interessa davvero.

**D — Perché alcuni modelli non hanno `class_weight`?**
KNN e Naïve Bayes non prevedono il parametro nella loro formulazione. Per loro lo
sbilanciamento si può gestire a livello di soglia decisionale, spostandola sotto 0.5.

**D — La multicollinearità tra le variabili economiche è un problema?**
Per Random Forest no; per la Regressione logistica rende instabili i coefficienti (li
rende difficili da interpretare), ma non ne compromette la capacità predittiva. La
regolarizzazione attenua il problema.

---

## 12. Concetti del corso utilizzati

| Concetto | Lezione |
|---|---|
| Attributi numerici vs nominali | 3 |
| Naïve Bayes, KNN, alberi, modelli lineari | 5–7 |
| Holdout, stratificazione, metriche, confusion matrix | 8 |
| Cross-validation, tuning (GridSearchCV) | 9 |
| One-hot encoding, scaling, imputazione | 11 |
| Data leakage | trasversale |
| Sbilanciamento delle classi e `class_weight` | trasversale |

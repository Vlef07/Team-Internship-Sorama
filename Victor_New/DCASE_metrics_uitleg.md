# DCASE Task 2 — AUC / pAUC (simpel uitgelegd)

## Wat wil de challenge?

- **AUC** en **pAUC** meten hoe goed je **anomaly scores** normale en abnormale testfragmenten **rangschikken**.
- **Hogere score = meer verdacht / meer anomalie** (zoals een grotere afstand tot je normale bibliotheek).
- **pAUC** gebruikt alleen het deel van de ROC-curve met **lage false-positive rate** (FPR ∈ [0, 0.1]); dat is `max_fpr=0.1` in `sklearn.metrics.roc_auc_score`.
- De **officiële score Ω** is een **harmonisch gemiddelde** over veel machine types, secties, en **source/target domeinen**. Daarvoor heb je per testclip **domain-labels** nodig; op de **development**-set in deze notebook zitten die vaak **niet** in dezelfde tabel — dan kun je alleen een **vereenvoudigde** samenvatting doen (bijv. harmonisch gemiddelde over machines).

## Victors code (CSV) — wat doet die?

```python
df['label'] = ...  # 1 als 'anomaly' in de naam, 0 als 'normal'
auc = metrics.roc_auc_score(df['label'], df['anomaly_score'])
pauc = metrics.roc_auc_score(df['label'], df['anomaly_score'], max_fpr=0.1)
```

- **Zelfde wiskunde** als: je hebt een lijst `y` (0/1) en een lijst `score` per rij; `roc_auc_score` sorteert impliciet en meet of anomalieën vaker **hogere** scores krijgen dan normalen.
- Een **CSV** is alleen een **bestandsformaat**: je kunt dezelfde kolommen als een `DataFrame` in het geheugen houden. CSV is handig om met je teammate te delen of naar de officiële evaluator te sturen.

## Oude KNN-cel — wat ging er mis?

We gebruiken nu: 

```python
denom = distances.max() if distances.max() > 0 else 1.0
y_scores = 1 - (distances / denom)
roc_auc = roc_auc_score(y_test, y_scores)
```

- `distances` is **hoger** bij **meer anomalie** (goed voor DCASE).
- `1 - distance/denom` maakt dat **anomalieën lagere** `y_scores` krijgen → **omgekeerde rangorde** t.o.v. wat `roc_auc_score` verwacht voor label **1 = anomalie** (hogere score = positiever).
- **Fix:** gebruik **`roc_auc_score(y_test, distances)`** direct (eventueel alleen schalen als je wilt, maar **monotone** transformaties veranderen AUC niet; teken omkeren wel).

## Waarom lijkt het op de DCASE-formule met H(·)?

De formule in de taskbeschrijving (paren vergelijken van scores van anomalieën vs normalen) is een **ranking**-definitie. Voor **binaire** labels is dat **equivalent** met de standaard ROC-AUC die `sklearn` berekent — mits je scores **consistent** zijn (hoger = meer anomalie).

## Source / target (domain)

Victor noemt **domain classification**: de officiële metriek splitst **source** en **target**. Die labels staan **niet** automatisch in je bestandsnaam; die komen uit de challenge-metadata / evaluator. Zonder die split: je AUC/pAUC per machine is nog steeds **vergelijkbaar** met Victors pipeline, maar **niet** identiek aan de volledige officiële Ω op alle domeinen.

# eat fine-tuning uitleg

## wat is er toegevoegd

in de notebook is een echte fine-tuning loop toegevoegd voor eat, zodat de gekozen feature mode (frozen, last_layer, lora, adapter) ook echt effect heeft op de embeddings.

## waarom dit nodig is

als je alleen `requires_grad` instelt, verandert het model nog niet.
er moet ook een optimizer stap zijn met:

- forward
- loss
- backward
- optimizer.step

zonder deze stappen blijven de embeddings vaak bijna gelijk tussen modi.

## nieuwe instellingen

de volgende instellingen zijn toegevoegd:

- `USE_EAT_FINETUNE`
- `EAT_FINETUNE_EPOCHS`
- `EAT_FINETUNE_LR`
- `EAT_FINETUNE_BATCH_SIZE`
- `EAT_FINETUNE_MAX_FILES`

## hoe de fine-tuning cel werkt

1. verzamelt alle train `.wav` bestanden uit de dev-dataset.
2. laadt audio met dezelfde preprocessing als je extractiepad.
3. past optioneel tse toe als `USE_TSE = True`.
4. maakt mel-input voor eat.
5. traint alleen parameters met `requires_grad = True`.
6. gebruikt een eenvoudige one-class objective:
   - embeddings in een batch worden richting batchcentrum getrokken.
7. zet het model na training terug op `eval()`.

## eerlijke vergelijking tussen modes

voor een eerlijke vergelijking:

1. kies een mode (`frozen`, `last_layer`, `lora`, `adapter`).
2. zet `USE_EAT_FINETUNE = True`.
3. run fine-tuning cel.
4. extraheer features opnieuw.
5. sla per mode apart op (bijv. `EAT_frozen`, `EAT_lora`).
6. run daarna pca en knn.

## belangrijke noot

als je dezelfde outputmap hergebruikt, kan skip-logic oude features blijven gebruiken.
dat maakt vergelijkingen oneerlijk.
gebruik daarom aparte outputnamen per mode of verwijder oude features voordat je opnieuw draait.

# Team-Internship-Sorama

## Initialization
Start een virtual environment.

Gebruik pip==24.0, en installeer de volgende packages:

`pip install scikit-learn h5py==3.10.0 numpy==1.26.3 omegaconf==2.0.6 pyarrow==15.0.0 scikit_learn==1.3.2 soundfile==0.12.1 timm==0.9.12 torch==2.1.2 torchaudio==2.1.2 torchsummary==1.5.1 tensorboardX==2.6.2.2 transformers==4.51.3 matplotlib `

Zorg er ook voor dat je de unilm/beats repo cloned voor het BEATs model. Clonen kan via https://github.com/microsoft/unilm/tree/master/beats. Zorg ook dat je `BEATs_iter3_plus_AS2M.pt` hebt download via de README in de unilm/beats repo en zet deze in deze git repo.

Voor het EAT model, clone de volgende git repo https://github.com/cwx-worst-one/EAT. 


Zorg ervoor dat je je data op de juiste manier opslaat. De data is te downloaden via: https://dcase.community/challenge2025/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring. Volg de volgende directory structure:

```text
data/
└── dcase2025t2/
    ├── dev_data/
    │   ├── processed/
    │   └── raw/
    │       └── {machine}/
    │           ├── test/
    │           ├── train/
    │           └── attributes_00.csv
    └── eval_data/
        ├── processed/
        └── raw/
            └── {machine}/
```
#### This repository contains codes and data for the article [Pull-back Geometry of Persistent Homology Encodings](https://arxiv.org/abs/2310.07073).

#### Folder structure
    .
    ├── Identifying-what-is-recognized      # Codes and data for section 4 "Identifying what is recognized"
    │   ├── Figure                          # Figures shown in section 4
    │   │   ├── align.png
    │   │   └── ...
    │   │
    │   ├── Experiments.ipynb               # Scripts to replicate the experiments in section 4
    │   ├── utilyze.py                      # Functions needed in the experiments
    │   └── pbn_perturb_func.py             # Function to compute average pull-back norm for perturbations
    │
    └── Selecting-hyperparameters           # Codes and data for section 5 "Selecting hyperparameters"
        ├── Data                            # Brain artery tree data
        │   ├── label.csv                   # Labels of brain artery trees
        │   └── data_pc.pkl                 # Vertices of brain artery trees
        │
        ├── Figure                          # Figures shown in section 5
        │   ├── correlation_cnn.png
        │   └── ...
        │
        ├── Experiments_1.ipynb             # Scripts to replicate the experiments in section 5.1
        ├── Experiments_2.ipynb             # Scripts to replicate the experiments in section 5.2
        └── utilyze.py                      # Functions needed in the experiments
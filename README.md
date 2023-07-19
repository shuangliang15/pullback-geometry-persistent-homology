# persistent-homology

## This repository contains codes and data for the article "Sensitivity Analysis of Persistent Homology via Pull-back Geometry".

#### Folder structure
    .
    ├── Identifying-what-is-recognized          # Codes and data for section 4 "Identifying what is recognized"
    │   ├── Figure                              # Figures shown in section 4
    │   │   ├── align.png
    │   │   └── ...
    │   ├── Experiments.ipynb                   # Main codes
    │   └── utilyze.py                          # Utilyze functions (e.g. functions used to compute Jacobian)
    └── Selecting-hyperparameters               # Codes and data for section 4 "Selecting hyperparameters"
        ├── Data                                # Brain artery tree data
        │   ├── label.csv                       # Labels of brain artery trees
        │   └── data_pc.pkl                     # Vertices of brain artery trees
        ├── Figure                              # Figures shown in section 5
        │   ├── correlation_cnn.png
        │   └── ...
        ├── Experiments.ipynb                   # Main codes
        └── utilyze.py                          # Utilyze functions
 

 Brain artery tree data is downloaded from https://gitlab.com/alexpieloch/PersistentHomologyAnalysisofBrainArteryTrees/-/tree/master/data/OriginalBrainTreeData.
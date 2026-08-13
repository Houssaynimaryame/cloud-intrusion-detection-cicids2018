# Données

## Ce qui est inclus dans ce dépôt (échantillons, légers)

Pour que ce dépôt reste facilement téléchargeable/uploadable sur GitHub, seuls des
**échantillons** du dataset sont inclus ici (les notebooks, eux, contiennent les vrais
résultats calculés sur le dataset complet — voir `notebooks/`) :

- `sample_flows.parquet` — échantillon stratifié (18 000 lignes, 6 000 par classe) issu du
  dataset complet après nettoyage et sélection de variables (36 features).
- `raw/` — petits échantillons (~6 000 lignes) des deux fichiers **bruts** (78 colonnes
  d'origine, avant nettoyage), utiles pour tester `notebooks/01_EDA.ipynb` rapidement.

Ces échantillons suffisent pour exécuter le pipeline de bout en bout (EDA, prétraitement,
entraînement, évaluation) et retrouver des résultats très proches de ceux du rapport.

## Obtenir le dataset complet (CSE-CIC-IDS2018)

Le dataset original est publié par le **Canadian Institute for Cybersecurity (CIC)** :

**Site officiel : https://www.unb.ca/cic/datasets/ids-2018.html**

Miroir également disponible sur AWS Open Data Registry et sur Kaggle (rechercher
*"CSE-CIC-IDS2018"*).

Ce projet utilise précisément deux fichiers (générés par CICFlowMeter, format Parquet) :

| Fichier | Date | Trafic |
|---|---|---|
| `DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet` | 20/02/2018 | Benign + DDoS attacks-LOIC-HTTP |
| `Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet` | 14/02/2018 | Benign + SSH-Bruteforce + FTP-BruteForce |

Pour reproduire exactement les chiffres du rapport (1 574 187 lignes, cf. README principal),
télécharger ces fichiers depuis le site officiel ci-dessus, les placer dans `data/raw/`
(remplaçant les échantillons), fusionner/nettoyer via `notebooks/01_EDA.ipynb`, puis exécuter
`notebooks/02_preprocessing_and_modeling.ipynb` ou `train.py` sur le résultat
(`data/df_selected.parquet`).

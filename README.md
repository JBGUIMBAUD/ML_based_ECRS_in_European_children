# Machine Learning based Mental, Cardiovascular and Respiratory Environmental-Clinical Risk Scores in European Children

## Overview
In this repository you will find the code used in our paper "Machine Learning based Mental, Cardiovascular and Respiratory Environmental-Clinical Risk Scores in European Children" publised in Nature Communications Medicine (2024).

## Repository Structure
This paper computed Environmental-Clinical risk scores (ECRS) for three main outcomes: p-factor, metabolic syndrome and lung function. You will find a folder corresponding to each outcome in which the corresponding ECRS using different methods.

### [p-factor/2sECRS_mental.ipynb]
- **Description**: In this notebook mental health (p-factor) ECRS are computed. Predictive performances are compared accross several models (XGBoost, Random Forest and LASSO) and finally Shapley values are extracted from the final ECRS to get local explanations.

### [mets/2sECRS_cardio.ipynb]
- **Description**: Computation of cardiovascular health ECRS. Again, predictive performances are compared accross several models (XGBoost, Random Forest and LASSO) and finally Shapley values are extracted from the final ECRS to get local explanations.

### [lung_function/2sECRS_resp.ipynb]
- **Description**: Computation of respiratory health ECRS. Again, predictive performances are compared accross several models (XGBoost, Random Forest and LASSO) and finally Shapley values are extracted from the final ECRS to get local explanations.

### [utils.py]
- **Description**: Python file where the main functions used in the above mentionned codebooks are implemented.

### [plot_figures.ipynb]
- **Description**: Code used to build the main figures displayed in the article.


## Requirements
Dependencies are listed in the requirements.txt file.


## Data
The raw data supporting this study are not shared within this repository. Those are available from the correspond author on request subject to ethical and legislative review. The "HELIX Data External Request Procedures" are available with the data inventory in http://www.projecthelix.eu/data-inventory. This document describes who can apply to the data and how, timings for approvals and other conditions to access and publication.


## License
Freely accessible under a [CC BY license](https://creativecommons.org/licenses/by/4.0/) (Creative Commons Attribution 4.0 International License).


## Acknowledgments
The authors would like to thank all the participating children, parents, practitioners, and researchers in the six countries who took part in this study. The Norwegian Mother, Father and Child Cohort Study is supported by the Norwegian Ministry of Health and Care Services and the Ministry of Education and Research. We also acknowledge the support of the Spanish Ministry of Science and Innovation to the EMBL partnership, the Centro de Excelencia Severo Ochoa, and the CERCA Programme / Generalitat de Catalunya. The CRG/UPF Proteomics Unit is part of the Spanish Infrastructure for Omics Technologies (ICTS OmicsTech) and it is supported by “Secretaria d’Universitats i Recerca del Departament d’Economia i Coneixement de la Generalitat de Catalunya” (2021SGR01225 and 2021SGR01563). 
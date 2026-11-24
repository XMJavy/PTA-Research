# PTA Hemodynamics Research

## Overview

This repository contains Python code for a research study of hemodynamic changes after percutaneous transluminal angioplasty (PTA). The study examines changes in blood flow (ΔQ) and resistance index (ΔRI), with the aim of identifying preoperative factors associated with postoperative outcomes.

## Analysis Approach

The analysis includes data quality assessment, univariable screening, multivariable regression with robust standard errors, and comparison of several machine-learning models. Patient-grouped cross-validation is used to reduce the risk of data leakage when multiple records come from the same patient. Model interpretation is supported by SHAP, partial dependence plots, and multi-model importance summaries.

## Data Privacy and Confidentiality

- No raw patient-level data are included in this repository.
- No patient names, medical record numbers, contact details, or other direct identifiers should be uploaded.
- Any data used with this code must be de-identified and accessed only with appropriate authorization.
- Clinical datasets should be stored separately in a secure location and must not be committed to GitHub.
- Reports or figures shared publicly should be reviewed to ensure that they contain no identifiable patient information.

## Reproducibility

The scripts require a locally stored, authorized dataset and the relevant Python dependencies. Because the clinical data are not distributed, only authorized researchers with data access can reproduce the complete analysis.

## Disclaimer

This code is intended for research purposes only and is not intended for clinical diagnosis or treatment decisions.

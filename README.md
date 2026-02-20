# RGS: ResNet50–GWO–SVM
**A Hybrid CNN–Metaheuristic Framework for Colorectal Polyp Classification**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Overview
RGS is a hybrid medical image classification framework that integrates deep learning and metaheuristic optimization to improve accuracy, stability, and efficiency in colorectal polyp classification.

The framework combines:
- **ResNet50** for deep feature extraction  
- **Grey Wolf Optimizer (GWO)** for wrapper-based feature selection  
- **Support Vector Machine (SVM)** for final classification  

By decoupling feature learning from classification, RGS reduces high-dimensional CNN feature spaces while improving generalization on imbalanced datasets.

---

## Architecture
The overall pipeline of the proposed RGS framework is illustrated below.

<p align="center">
  <img src="rgs_framework.pdf" width="750">
</p>

---

## Algorithm Details

### Grey Wolf Optimizer (GWO)
GWO is employed as a binary wrapper-based feature selection method. Each wolf represents a candidate feature subset, and the fitness function balances classification error and feature compactness.

**Key parameters used in this implementation:**

| Parameter | Symbol | Value |
|--------|--------|------|
| Population size | N | 30 |
| Maximum iterations | t_max | 10 |
| Feature dimension | D | 2048 |
| Fitness weight | α | 0.99 |
| Threshold | τ | 0.5 |
| Encoding | — | Binary |

**Fitness function:**

F(s) = α · error(s) + (1 − α) · (number of selected features / total features)

---

## Datasets

| Dataset | Images | Classes |
|------|------|------|
| PolypsSet | 35,981 | 2 |
| CP-CHILD-A | 8,000 | 2 |
| CP-CHILD-B | 1,500 | 2 |

All experiments use stratified 10-fold cross-validation to preserve class distributions.

---

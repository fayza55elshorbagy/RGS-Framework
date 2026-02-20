# RGS-Framework
RGS is a hybrid deep learning–metaheuristic framework for medical image classification. It combines pretrained CNN feature extraction with optimization-driven feature selection and margin-based classification, achieving compact representations and improved generalization.

# RGS integrates:
1)ResNet50 → deep feature extraction
2)Grey Wolf Optimizer (GWO) → wrapper-based feature selection
3)Support Vector Machine (SVM) → final classification

# Designed to:
-Reducing feature complexity while preserving model accuracy
-Improving computational efficiency

# Pipeline Description
-Image preprocessing (resize, normalize)
-Feature extraction using ResNet50 (2048-D vectors)
-Feature selection using GWO (binary wrapper selection)
-Classification using SVM
-Evaluation with stratified 10-fold cross-validation

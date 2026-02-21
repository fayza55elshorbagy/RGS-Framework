import os
# === SETUP RESULTS DIRECTORY ===
results_dir = "/content/drive/MyDrive/res_folder"
os.makedirs(results_dir, exist_ok=True)

# === IMPORT LIBRARIES ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# === CLASSIFIERS ===
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, cohen_kappa_score,
    roc_curve
)

import glob

# === PARAMETERS ===
ModelName = "GWO"
CFName = "SVM"     # choose: "LR", "DT", "NB", "KNN", "SVM", "RandomForest"
dbName = 'Database.csv'

N, T, k = 10, 10, 5

# === LOAD DATA ===
ds = pd.read_csv(dbName)
X = MinMaxScaler().fit_transform(ds.iloc[:, :-1].values)
y = ds.iloc[:, -1].astype(int).values

# === FEATURE SELECTION ===
print("Performing feature selection...")
from FS.gwo import jfs   # GWO optimizer

opts = {'k': k, 'fold': None, 'N': N, 'T': T, 'CFName': CFName}

start_time = time()
fmdl = jfs(X, y, opts)
fs_time = time() - start_time

selected_features = fmdl['sf']
X_reduced = X[:, selected_features]

# === SAVE CONVERGENCE CURVE ===
curve = fmdl['c'].reshape(-1)
generations = np.arange(1, len(curve)+1)

plt.figure(figsize=(6, 5))
plt.plot(generations, curve, marker='o')
plt.title(f'{ModelName} Convergence Curve')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'convergence_curve.pdf'))
plt.show()

# === STRATIFIED 10-FOLD CROSS-VALIDATION ===
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for i, (train_idx, test_idx) in enumerate(skf.split(X_reduced, y), 1):
    fold_pred_path = os.path.join(results_dir, f"fold_{i}.csv")
    fold_metric_path = os.path.join(results_dir, f"fold_{i}_metrics.csv")

    if os.path.exists(fold_pred_path) and os.path.exists(fold_metric_path):
        print(f"⏭️ Fold {i} already computed, skipping...")
        continue

    print(f"\n▶️ Fold {i}")
    X_train, X_test = X_reduced[train_idx], X_reduced[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # === CHOOSE CLASSIFIER ===
    if CFName == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=k)
    elif CFName == "SVM":
        from sklearn.svm import SVC
        model = SVC(kernel="linear", probability=True)
    elif CFName == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    else:
        raise ValueError("Unsupported classifier: choose LR, DT, NB, KNN, SVM, RandomForest")

    # === TRAIN AND PREDICT ===
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    # === METRICS ===
    acc = np.mean(y_test == y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    spec = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    kappa = cohen_kappa_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score)

    print(f"Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, "
          f"Specificity: {spec:.4f}, Kappa: {kappa:.4f}, AUC: {roc_auc:.4f}")

    # === SAVE PREDICTIONS ===
    fold_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_score": y_score
    })
    fold_df.to_csv(fold_pred_path, index=False)

    # === SAVE METRICS ===
    metrics_df = pd.DataFrame({
        "Accuracy": [acc],
        "F1": [f1],
        "Precision": [prec],
        "Recall": [rec],
        "Specificity": [spec],
        "Kappa": [kappa],
        "ROC_AUC": [roc_auc]
    })
    metrics_df.to_csv(fold_metric_path, index=False)

    print(f"✅ Saved predictions to {fold_pred_path}")
    print(f"✅ Saved metrics to {fold_metric_path}")

# === AGGREGATE RESULTS ===
fold_files = sorted(glob.glob(os.path.join(results_dir, "fold_*_metrics.csv")))
print(f"\n🔁 Found {len(fold_files)} saved folds. Aggregating...")

all_y_true, all_y_pred, all_y_scores = [], [], []
all_metrics = []

for i in range(1, len(fold_files)+1):
    pred_path = os.path.join(results_dir, f"fold_{i}.csv")
    metric_path = os.path.join(results_dir, f"fold_{i}_metrics.csv")

    if not os.path.exists(pred_path) or not os.path.exists(metric_path):
        continue

    df_pred = pd.read_csv(pred_path)
    df_metrics = pd.read_csv(metric_path)

    all_y_true.extend(df_pred["y_true"])
    all_y_pred.extend(df_pred["y_pred"])
    all_y_scores.extend(df_pred["y_score"])
    all_metrics.append(df_metrics.iloc[0])

# === SUMMARY METRICS ===
metric_df = pd.DataFrame(all_metrics)
metric_df.to_csv(os.path.join(results_dir, "fold_metrics.csv"), index=False)

metric_mean = metric_df.mean()
metric_std = metric_df.std()
summary_df = pd.DataFrame({
    "Mean": metric_mean,
    "Std": metric_std
})
summary_df.to_csv(os.path.join(results_dir, "metrics_summary.csv"))
print("\n📊 Saved metrics summary.")

# === CONFUSION MATRIX ===
agg_cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(agg_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.pdf'))
plt.show()

# === ROC CURVE ===
fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
roc_auc = roc_auc_score(all_y_true, all_y_scores)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'roc_curve.pdf'))
plt.show()

print(f"\n✅ Feature selection took {fs_time:.2f} seconds")
print("\n✅ Confusion Matrix (Aggregated):\n", agg_cm)
print("\n✅ ROC Curve AUC: {:.4f}".format(roc_auc))
print("\n✅ Average Metrics Across Folds:\n", summary_df)

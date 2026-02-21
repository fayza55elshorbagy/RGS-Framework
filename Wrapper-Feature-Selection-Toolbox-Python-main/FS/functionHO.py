import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# =============================
# CROSS-VALIDATION ERROR RATE
# =============================
def error_rate(X, y, x, opts):

    k      = opts.get('k', 5)
    CFName = opts.get('CFName', 'SVM')
    cv     = opts.get('cv', 10)

    # reduce dataset
    X_selected = X[:, x == 1]

    if X_selected.shape[1] == 0:
        return 1   # no features selected → worst score

    # CLASSIFIER CHOICE
    if CFName == "KNN":
        mdl = KNeighborsClassifier(n_neighbors=k)
    elif CFName == "SVM":
        mdl = SVC(kernel="linear")
    elif CFName == "RandomForest":
        mdl = RandomForestClassifier()
    elif CFName == "LR":
        mdl = LogisticRegression(max_iter=1000)
    elif CFName == "DT":
        mdl = DecisionTreeClassifier()
    elif CFName == "NB":
        mdl = GaussianNB()
    else:
        raise ValueError("Unsupported classifier: " + CFName)

    acc = cross_val_score(mdl, X_selected, y, cv=cv).mean()
    return 1 - acc


# =============================
# GWO OBJECTIVE FUNCTION
# =============================
def Fun(xtrain, ytrain, x, opts):

    alpha = opts.get("alpha", 0.99)
    beta  = opts.get("beta", 0.01)

    max_feat = len(x)
    num_feat = np.sum(x == 1)

    if num_feat == 0:
        return 1

    err = error_rate(xtrain, ytrain, x, opts)
    fitness = alpha * err + beta * (num_feat / max_feat)

    return fitness

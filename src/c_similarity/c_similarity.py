from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# ---- Step 1: Construct training data for f_C ----

# Assume D_pos and D_neg are lists of sets of word indices
# You already trained the merger: merger.fit(D_pos, D_neg, V_size)

# get feature vectors for each segment in D_train
Z_pos, Z_neg = run_pipeline()
# Z_pos = merger.transform_many(D_pos)  # shape: (n_pos, nf)
# Z_neg = merger.transform_many(D_neg)  # shape: (n_neg, nf)

X_train = np.vstack([Z_pos, Z_neg])
y_train = np.array([1] * len(Z_pos) + [0] * len(Z_neg))

# ---- Step 2: Train classifier f_C ----

# You can use logistic regression or random forest
clf = LogisticRegression(solver="liblinear", random_state=42)
# OR: clf = RandomForestClassifier(n_estimators=100, random_state=42)

clf.fit(X_train, y_train)

# ---- Step 3: Compute C-similarity for new segments ----


def compute_c_similarity(segments, merger, clf, threshold=0.1):
    """
    segments: list of sets of word indices (i.e., W(s_t) for each segment)
    merger: trained GreedyFeatureMerger
    clf: trained classifier (f_C)
    threshold: ψ_min

    Returns:
        list of (ψ_A(s_t), is_active)
    """
    Z_test = merger.transform_many(segments)
    probs = clf.predict_proba(Z_test)[:, 1]  # probability of class 1 (target app)

    results = []
    for score in probs:
        is_active = score >= threshold
        results.append((score, is_active))
    return results

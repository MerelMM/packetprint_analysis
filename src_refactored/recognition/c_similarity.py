from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import pickle
import os


def compute_c_similarity(app_key, fvs, threshold=None, model_path="data/classifiers/"):
    """
    Computes C-similarity scores using a pretrained classifier.

    Args:
        app_key (str): Identifier used to locate the trained model.
        fvs (List[np.ndarray] or np.ndarray): Compressed feature vectors for segments.
        model_path (str): Path to the directory containing trained models.

    Returns:
        np.ndarray: Array of predicted probabilities (C-similarity scores).
    """
    # Load trained classifier
    model_file = os.path.join(model_path, f"{app_key}_c_similarity_model.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No classifier found at {model_file}")

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Convert to numpy array if not already
    X = np.array(fvs)

    # Compute predicted probabilities for class 1 (positive)
    c_similarity_scores = model.predict_proba(X)[:, 1]

    if threshold is not None:
        return (c_similarity_scores >= threshold).astype(int)

    return c_similarity_scores


def train_c_similarity_classifier(
    app_key, fvs, lfm_labels, save_path="data/classifiers/"
):
    """
    Trains a logistic regression classifier for C-similarity classification.

    Args:
        app_key (str): Identifier for the app, used in saved file name.
        fvs (List[np.ndarray]): Compressed feature vectors.
        lfm_labels (List[int]): Binary labels corresponding to each segment.
        save_path (str): Path to directory for saving the trained model.

    Returns:
        model (LogisticRegression): Trained classifier.
    """

    # Make sure save path exists
    os.makedirs(save_path, exist_ok=True)

    # Convert to numpy arrays if not already
    X = np.array(fvs)
    y = np.array(lfm_labels)

    # Train logistic regression classifier
    model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
    model.fit(X, y)

    # Evaluate using 5-fold CV
    scores = cross_val_score(model, X, y, cv=5)
    print(
        f"[{app_key}] Logistic Regression Accuracy: {scores.mean():.3f} ± {scores.std():.3f}"
    )

    # Optionally print full classification report
    y_pred = model.predict(X)
    print(classification_report(y, y_pred, target_names=["neg", "pos"]))

    # Save model
    model_path = os.path.join(save_path, f"{app_key}_c_similarity_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved C-similarity classifier for {app_key} to {model_path}")

    return model


def compare_methods(fvs, lfm_labels):
    """
    Compares Logistic Regression and Random Forest using 5-fold cross-validation.

    Args:
        fvs (array-like): Feature vectors (samples × features)
        lfm_labels (array-like): Binary labels (0 or 1)

    Prints:
        Mean and std of cross-validation accuracy scores
    """
    lr = LogisticRegression(
        penalty="l2", solver="liblinear", max_iter=1000, class_weight="balanced"
    )
    rf = RandomForestClassifier(
        n_estimators=30, max_depth=6, random_state=42, class_weight="balanced"
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_scores = cross_val_score(lr, fvs, lfm_labels, cv=cv)
    rf_scores = cross_val_score(rf, fvs, lfm_labels, cv=cv)

    print(
        "Logistic Regression: {:.3f} ± {:.3f}".format(lr_scores.mean(), lr_scores.std())
    )
    print(
        "Random Forest:      {:.3f} ± {:.3f}".format(rf_scores.mean(), rf_scores.std())
    )

    """ 
    Result: 
    Logistic Regression:    1.000 ± 0.000
    Random Forest:          1.000 ± 0.000
    """

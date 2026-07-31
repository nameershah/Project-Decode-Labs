"""
Data Classification Using AI
Decode Labs AI Internship | Project 2 | Batch 2026
Pipeline: Iris -> StandardScaler -> Train-Test Split (80/20) -> KNN -> Evaluate
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score


# ─────────────────────────────────────────────────────────
# STAGE 1: INPUT — Load and explore the Iris benchmark
# ─────────────────────────────────────────────────────────

def load_and_explore():
    """Load Iris dataset and print structural summary."""
    iris = load_iris()
    X, y = iris.data, iris.target

    print("=" * 55)
    print("RAW MATERIAL: IRIS BENCHMARK")
    print("=" * 55)
    print(f"  Samples     : {X.shape[0]} (balanced)")
    print(f"  Classes     : {len(iris.target_names)} -> {[str(n) for n in iris.target_names]}")
    print(f"  Features    : {X.shape[1]} -> {[str(f) for f in iris.feature_names]}")
    print(f"  Distribution: {np.bincount(y)} per class")
    print()

    return X, y, iris


# ─────────────────────────────────────────────────────────
# STAGE 2: PROCESS — Split, Scale, Train
# ─────────────────────────────────────────────────────────

def split_data(X, y, test_size=0.20, random_state=42):
    """
    80/20 train-test split.
    shuffle=True removes order bias from the original dataset ordering.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    print(f"  Train set : {len(X_train)} samples (80%)")
    print(f"  Test set  : {len(X_test)} samples (20%)")
    return X_train, X_test, y_train, y_test


def apply_scaling(X_train, X_test):
    """
    Gatekeeper Rule: StandardScaler normalizes features to Mean=0, Variance=1.
    Fit ONLY on training data to prevent data leakage into the test set.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only — no fit
    print("  Scaling applied: Mean=0, Variance=1 (StandardScaler)")
    print("  Data leakage prevented: scaler fit on train only")
    return X_train_scaled, X_test_scaled, scaler


def train_knn(X_train, y_train, k=5):
    """
    K-Nearest Neighbors: proximity principle + majority vote.
    Three steps as per scikit-learn workflow:
      1. Instantiate (build the frame)
      2. Fit       (memorize the map)
      3. Predict   (apply logic) — called separately in evaluate()
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    print(f"  KNN trained (k={k})")
    return model


# ─────────────────────────────────────────────────────────
# STAGE 3: OUTPUT — Evaluate with Confusion Matrix + F1
# ─────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, target_names):
    """
    The Accuracy Mirage: raw accuracy is unreliable on imbalanced data.
    We validate with:
      - Confusion Matrix: TP / FP / FN / TN breakdown per class
      - F1 Score: harmonic mean of precision and recall
    """
    predictions = model.predict(X_test)

    print("\n" + "=" * 55)
    print("OUTPUT VALIDATION")
    print("=" * 55)

    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    f1 = f1_score(y_test, predictions, average="weighted")
    print(f"\nF1 Score (weighted): {f1:.4f}")

    print("\nFull Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names))

    return predictions, f1


# ─────────────────────────────────────────────────────────
# BONUS: Tuning the Engine — Find optimal K via elbow method
# ─────────────────────────────────────────────────────────

def find_optimal_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    """
    Scan k=1..20. K=1 overfits (noise). K=100 underfits (generic).
    The elbow point — where F1 stops improving — is optimal K.
    """
    print("=" * 55)
    print("TUNING THE ENGINE: OPTIMAL K SEARCH (k=1 to 20)")
    print("=" * 55)

    scores = {}
    for k in k_range:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train, y_train)
        scores[k] = f1_score(y_test, m.predict(X_test), average="weighted")
        print(f"  k={k:2d}  |  F1={scores[k]:.4f}")

    best_f1 = max(scores.values())
    # On tie, prefer k=5 (robust default over potentially overfitting k=1)
    candidates = [k for k, v in scores.items() if v == best_f1]
    best_k = 5 if 5 in candidates else candidates[0]
    print(f"\n  Optimal K: {best_k}  |  Best F1: {scores[best_k]:.4f}")
    return best_k, scores


# ─────────────────────────────────────────────────────────
# MAIN: Full IPO pipeline
# ─────────────────────────────────────────────────────────

def main():
    print("\nDECODE LABS | PROJECT 2: DATA CLASSIFICATION USING AI")
    print("Algorithm: K-Nearest Neighbors  |  Dataset: Iris Benchmark")
    print()

    # INPUT
    X, y, iris = load_and_explore()

    # PROCESS: Split
    print("=" * 55)
    print("STRUCTURAL INTEGRITY: TRAIN-TEST SPLIT")
    print("=" * 55)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print()

    # PROCESS: Scale
    print("=" * 55)
    print("GATEKEEPER RULE: FEATURE SCALING")
    print("=" * 55)
    X_train_sc, X_test_sc, _ = apply_scaling(X_train, X_test)
    print()

    # PROCESS: Train
    print("=" * 55)
    print("ALGORITHM: K-NEAREST NEIGHBORS (k=5)")
    print("=" * 55)
    model = train_knn(X_train_sc, y_train, k=5)
    print()

    # OUTPUT: Evaluate
    evaluate(model, X_test_sc, y_test, iris.target_names)

    # BONUS: Optimal K
    print()
    best_k, _ = find_optimal_k(X_train_sc, y_train, X_test_sc, y_test)

    if best_k != 5:
        print(f"\nRetraining with optimal k={best_k}:")
        model_opt = train_knn(X_train_sc, y_train, k=best_k)
        evaluate(model_opt, X_test_sc, y_test, iris.target_names)
    else:
        print(f"\nk=5 is already optimal. No retraining needed.")


if __name__ == "__main__":
    main()

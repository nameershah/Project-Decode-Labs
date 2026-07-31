# Data Classification Using AI
Decode Labs AI Internship | Project 2 | Batch 2026

## Overview
A supervised learning pipeline that classifies Iris flower species using K-Nearest Neighbors. Implements the full ML lifecycle across all three IPO stages: data loading and scaling, train-test split, model training, and validation via confusion matrix and weighted F1 score. Moves beyond heuristic rule-writing — the machine derives the logic from labeled history.

## How to Run
```
pip install scikit-learn numpy
python classification.py
```
No other dependencies. Standard Python 3 only.

## Pipeline (IPO Framework)
| Stage | Step | Detail |
|-------|------|--------|
| Input | Dataset | Iris benchmark — 150 samples, 3 classes, 4 features |
| Input | Feature scaling | StandardScaler: Mean=0, Variance=1 |
| Process | Train-Test Split | 80% train / 20% test, shuffle=True (removes order bias) |
| Process | KNN Classifier | k=5, proximity principle + majority vote |
| Process | K Tuning | Elbow method over k=1..20, prefers k=5 on tie |
| Output | Confusion Matrix | TP / FP / FN / TN per class |
| Output | F1 Score | Harmonic mean of precision and recall (weighted) |

## Architecture
| Component | Implementation |
|-----------|---------------|
| Dataset loader | `sklearn.datasets.load_iris` |
| Feature scaling | `sklearn.preprocessing.StandardScaler` |
| Train-test split | `sklearn.model_selection.train_test_split` (shuffle=True) |
| Classifier | `sklearn.neighbors.KNeighborsClassifier` |
| Evaluation | `confusion_matrix` + `f1_score` (weighted) |
| K tuning | Elbow method over k=1..20 |

## Key Concepts Applied
- **The Logic Skeleton**: supervised learning replaces hand-written if/else rules by learning decision boundaries from labeled data.
- **The Gatekeeper Rule**: StandardScaler is fit only on training data, preventing test-set information from leaking into preprocessing.
- **Structural Integrity**: shuffle before split removes order bias that exists in the original Iris dataset ordering.
- **Proximity Principle**: KNN classifies by majority vote among the k nearest neighbors in feature space.
- **The Accuracy Mirage**: raw accuracy misleads on imbalanced data; weighted F1 score gives the honest picture.
- **Tuning the Engine**: k=1 overfits noise; high k underfits; the elbow method finds the balanced optimum.

## Example Output
```
DECODE LABS | PROJECT 2: DATA CLASSIFICATION USING AI
Algorithm: K-Nearest Neighbors  |  Dataset: Iris Benchmark

RAW MATERIAL: IRIS BENCHMARK
  Samples     : 150 (balanced)
  Classes     : 3 -> ['setosa', 'versicolor', 'virginica']
  Features    : 4 -> ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

  Train set : 120 samples (80%)
  Test set  : 30 samples (20%)

  Scaling applied: Mean=0, Variance=1 (StandardScaler)
  KNN trained (k=5)

Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

F1 Score (weighted): 1.0000
```

## About
No external dependencies beyond scikit-learn and numpy.

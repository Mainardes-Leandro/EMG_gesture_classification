"""
train.py
--------
Train and compare classifiers for EMG gesture recognition.
Outputs metrics and saves results to /results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Classifiers ───────────────────────────────────────────────────────────────

def build_classifiers(n_classes: int) -> dict:
    """Return a dict of sklearn-compatible pipelines."""
    clfs = {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                          decision_function_shape="ovr", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=None,
                                              n_jobs=-1, random_state=42))
        ]),
    }
    if XGBOOST_AVAILABLE:
        clfs["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    XGBClassifier(n_estimators=200, max_depth=6,
                                     learning_rate=0.1, use_label_encoder=False,
                                     eval_metric="mlogloss", n_jobs=-1,
                                     random_state=42))
        ])
    return clfs


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_and_evaluate(X_train, y_train, X_val, y_val,
                       X_test, y_test, gesture_names: list = None):
    """
    Train all classifiers, evaluate on val and test sets.
    Saves metrics JSON and plots to /results.

    Returns a dict with results per classifier.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val   = le.transform(y_val)
    y_test  = le.transform(y_test)

    n_classes = len(np.unique(y_train))
    clfs = build_classifiers(n_classes)
    results = {}

    for name, pipeline in clfs.items():
        print(f"\n[{name}] Training...")
        pipeline.fit(X_train, y_train)

        val_pred  = pipeline.predict(X_val)
        test_pred = pipeline.predict(X_test)

        val_acc   = accuracy_score(y_val,  val_pred)
        test_acc  = accuracy_score(y_test, test_pred)
        test_f1   = f1_score(y_test, test_pred, average="macro")

        print(f"  Val  accuracy : {val_acc:.4f}")
        print(f"  Test accuracy : {test_acc:.4f}")
        print(f"  Test F1 macro : {test_f1:.4f}")

        results[name] = {
            "val_accuracy":  round(val_acc,  4),
            "test_accuracy": round(test_acc, 4),
            "test_f1_macro": round(test_f1,  4),
            "pipeline":      pipeline,
            "test_pred":     test_pred,
        }

    # Save metrics to JSON
    metrics = {k: {m: v for m, v in v.items()
                   if m not in ("pipeline", "test_pred")}
               for k, v in results.items()}
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Plots
    _plot_comparison(results)
    _plot_confusion_matrix(results, y_test, gesture_names)

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_comparison(results: dict):
    """Bar chart comparing classifiers on val and test accuracy."""
    names     = list(results.keys())
    val_accs  = [results[n]["val_accuracy"]  for n in names]
    test_accs = [results[n]["test_accuracy"] for n in names]
    f1s       = [results[n]["test_f1_macro"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, val_accs,  width, label="Val Accuracy",  color="#4C72B0")
    ax.bar(x,         test_accs, width, label="Test Accuracy", color="#55A868")
    ax.bar(x + width, f1s,       width, label="Test F1 Macro", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Classifier Comparison — NinaPro DB5 Gesture Recognition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "classifier_comparison.png"), dpi=150)
    plt.close()
    print("\n[Saved] results/classifier_comparison.png")


def _plot_confusion_matrix(results: dict, y_test: np.ndarray,
                            gesture_names: list = None):
    """Save one confusion matrix per classifier."""
    for name, res in results.items():
        cm = confusion_matrix(y_test, res["test_pred"])
        labels = gesture_names if gesture_names else np.unique(y_test)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=len(labels) <= 20, fmt="d",
                    xticklabels=labels, yticklabels=labels,
                    cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {name}")
        plt.tight_layout()

        fname = name.replace(" ", "_").replace("(", "").replace(")", "") + "_cm.png"
        plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
        plt.close()
        print(f"[Saved] results/{fname}")

"""
ECE 5494 AI Neural Network Project
Neural Network-Based Early Risk Detection for Collegiate Student-Athletes

Run from the project folder with:
    python run_project.py

This script loads the prepared final dataset, trains a Logistic Regression
baseline and an MLP neural network classifier, evaluates both models, and
saves comparison metrics and figures in the outputs/ folder.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "final_student_athlete_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "at_risk" not in df.columns:
        raise ValueError("The dataset must contain an 'at_risk' target column.")
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare X/y while removing obvious label leakage and ID columns."""
    y = pd.to_numeric(df["at_risk"], errors="coerce")
    valid_rows = y.notna()
    df = df.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)

    # Columns that directly reveal the target or identify records.
    leakage_exact = {
        "at_risk",
        "academic_score_raw",
        "previous_gpa",
        "gpa",
        "g1",
        "g2",
        "g3",
        "math score",
        "reading score",
        "writing score",
        "math_score",
        "reading_score",
        "writing_score",
        "exam_score",
        "final_grade",
        "previous_scores",
        "target",
        "grade",
        "result",
        "pass_fail",
        "pass/fail",
        "student id",
        "student_id",
        "id",
        "name",
    }

    drop_cols = []
    for col in df.columns:
        clean = str(col).lower().strip()
        if clean in leakage_exact or clean.isdigit() or "curricular units 1st sem (grade)" == clean or "curricular units 2nd sem (grade)" == clean:
            drop_cols.append(col)

    X = df.drop(columns=drop_cols, errors="ignore")

    # Convert object columns that are mostly numeric.
    for col in X.columns:
        if X[col].dtype == "object":
            converted = pd.to_numeric(X[col], errors="coerce")
            if converted.notna().mean() >= 0.80:
                X[col] = converted

    # Remove very high-cardinality categorical columns, if any.
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    high_cardinality = [c for c in cat_cols if X[c].nunique(dropna=True) > 50]
    X = X.drop(columns=high_cardinality, errors="ignore")

    return X, y, drop_cols, high_cardinality


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
    }

    report_path = OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"{name} ROC Curve")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_roc_curve.png", dpi=150)
    plt.close()

    return metrics


def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    X, y, dropped, high_card = prepare_features(df)
    print(f"Feature matrix shape after leakage/ID cleanup: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().to_string()}")
    print(f"Dropped {len(dropped)} leakage/ID columns.")
    if high_card:
        print(f"Dropped high-cardinality categorical columns: {high_card}")

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining Logistic Regression baseline...")
    log_reg = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )
    log_reg.fit(X_train, y_train)

    print("Training MLP neural network classifier...")
    mlp = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    max_iter=200,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    mlp.fit(X_train, y_train)

    print("\nEvaluating models...")
    results = [
        evaluate_model("Logistic Regression", log_reg, X_test, y_test),
        evaluate_model("MLP Neural Network", mlp, X_test, y_test),
    ]

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "model_comparison_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nModel Comparison Results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved results and figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

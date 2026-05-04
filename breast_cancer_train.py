import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Paths & settings — where files live, test-set fraction, and reproducibility seed
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "breast_cancer_wisconsin", "data.csv")
MODEL_PATH = "breast_cancer_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------------------------------------------------------
# Load data — read CSV and fail fast if the dataset is missing
# -----------------------------------------------------------------------------
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}.\n"
        "Download the Kaggle dataset and copy data.csv into that folder."
    )

df = pd.read_csv(DATA_PATH)

# -----------------------------------------------------------------------------
# Feature matrix & labels — drop IDs, encode diagnosis (B/M → 0/1), keep numeric columns only
# -----------------------------------------------------------------------------
for col in ("id", "Unnamed: 0"):
    if col in df.columns:
        df = df.drop(columns=[col])

if "diagnosis" not in df.columns:
    raise ValueError("Expected a 'diagnosis' column in the CSV.")

y = df["diagnosis"].map({"M": 1, "B": 0})
if y.isna().any():
    raise ValueError("diagnosis should only contain M and B.")

X = df.drop(columns=["diagnosis"])

X = X.select_dtypes(include=[np.number])

print("Number of samples:", len(df))
print("Number of features:", X.shape[1])
print("Classes: 0=benign (B), 1=malignant (M)")

# -----------------------------------------------------------------------------
# Train / test split — hold out a stratified test set for honest performance estimates
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# -----------------------------------------------------------------------------
# Model — scale features then fit logistic regression on the training split
# -----------------------------------------------------------------------------
model = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        (
            "classifier",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
    ]
)

model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# Metrics — accuracy, precision, recall, F1, and full classification report on test data
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMODEL PERFORMANCE (on held-out test set)")
print("------------------------------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}  (of predicted malignant, how many were correct)")
print(f"Recall    : {recall:.4f}  (of real malignant cases, how many we caught)")
print(f"F1-score  : {f1:.4f}")

target_names = ["benign (B)", "malignant (M)"]
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# -----------------------------------------------------------------------------
# Confusion matrix — plot true vs predicted counts and save as an image
# -----------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=target_names,
    yticklabels=target_names,
    ylabel="True label",
    xlabel="Predicted label",
    title="Confusion matrix (breast cancer)",
)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

fig.tight_layout()
fig_path = "breast_cancer_confusion_matrix.png"
plt.savefig(fig_path, dpi=150)
print(f"\nSaved confusion matrix figure to {fig_path}")
plt.show()

# -----------------------------------------------------------------------------
# Save artifact — persist the fitted pipeline and feature column names for prediction
# -----------------------------------------------------------------------------
artifact = {
    "model": model,
    "feature_columns": list(X.columns),
}
joblib.dump(artifact, MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")
print("Use breast_cancer_predict.py to run predictions on new rows.")

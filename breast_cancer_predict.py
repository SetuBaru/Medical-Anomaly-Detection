import os
import joblib
import pandas as pd

# -----------------------------------------------------------------------------
# Paths — saved model and input CSV used for demo predictions
# -----------------------------------------------------------------------------
MODEL_PATH = "breast_cancer_model.joblib"
DATA_PATH = os.path.join("data", "breast_cancer_wisconsin", "data.csv")


# -----------------------------------------------------------------------------
# load_artifact — load the joblib bundle (model + feature column list)
# -----------------------------------------------------------------------------
def load_artifact(path=MODEL_PATH):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Train first — missing {path}")
    return joblib.load(path)


# -----------------------------------------------------------------------------
# predict_dataframe — attach malignant probability and thresholded class labels
# -----------------------------------------------------------------------------
def predict_dataframe(model, feature_columns, df):
    X = df[feature_columns]
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    out = df.copy()
    out["prob_malignant"] = proba
    out["predicted_malignant"] = pred
    return out


def main():
    # -------------------------------------------------------------------------
    # Load trained model from disk
    # -------------------------------------------------------------------------
    artifact = load_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    # -------------------------------------------------------------------------
    # Load CSV and align columns with training (drop id/label columns, keep features)
    # -------------------------------------------------------------------------
    if not os.path.isfile(DATA_PATH):
        print(f"No {DATA_PATH} — edit DATA_PATH or add your own CSV with the same columns.")
        return

    df = pd.read_csv(DATA_PATH)
    for col in ("id", "Unnamed: 0", "diagnosis"):
        if col in df.columns:
            df = df.drop(columns=[col])

    df = df[feature_columns]
    sample = df.head(5)
    # -------------------------------------------------------------------------
    # Predict — run the pipeline on a small sample and print probabilities / classes
    # -------------------------------------------------------------------------
    result = predict_dataframe(model, feature_columns, sample)

    # -------------------------------------------------------------------------
    # Output — explain columns printed for the demo rows
    # -------------------------------------------------------------------------
    print("Demo predictions (first 5 rows of the dataset, without using true labels):\n")
    print(result[["prob_malignant", "predicted_malignant"]].to_string())
    print("\nprob_malignant = model's estimated chance the tumor is malignant (M).")
    print("predicted_malignant: 0 = benign, 1 = malignant (threshold 0.5).")


if __name__ == "__main__":
    main()

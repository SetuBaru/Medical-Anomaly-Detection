import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import seaborn as sns

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("model.h5")
print("✅ Model loaded")

# =========================
# LOAD TEST DATA
# =========================

test_dir = "dataset/Testing"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # IMPORTANT for correct metrics
)

class_names = list(test_data.class_indices.keys())
print("Classes:", class_names)

# =========================
# PREDICTIONS
# =========================

y_true = test_data.classes
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)

# =========================
# BASIC METRICS
# =========================

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("\n📊 MODEL PERFORMANCE")
print("---------------------")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# =========================
# CLASSIFICATION REPORT
# =========================

print("\n📄 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# =========================
# OPTIONAL: LOSS ON TEST SET
# =========================

loss, acc = model.evaluate(test_data, verbose=1)

print("\n📉 Test Loss:", loss)
print("📊 Test Accuracy:", acc)
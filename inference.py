import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("model.h5")
print("✅ Model loaded")

# IMPORTANT: must match training order
class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

IMG_SIZE = (128, 128)

# =========================
# PREDICTION FUNCTION
# =========================

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    x = np.array(img) / 255.0
    x = x.reshape(1, 128, 128, 3)

    pred = model.predict(x)
    idx = np.argmax(pred)

    label = class_names[idx]
    confidence = pred[0][idx] * 100

    print(f"{confidence:.2f}% → {label}")

    plt.imshow(img)
    plt.title(label)
    plt.axis("off")
    plt.show()

# =========================
# TEST
# =========================

predict_image("dataset/Testing/no_tumor/no_tumor31.jpg")
predict_image("dataset/Testing/glioma/glioma3.png")
predict_image("dataset/Testing/meningioma/meningioma7.png")
predict_image("dataset/Testing/pituitary/pituitary104.png")
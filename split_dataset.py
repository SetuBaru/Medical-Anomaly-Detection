import os
import shutil
import random

# =========================
# CONFIG
# =========================

SOURCE_DIR = "archive"      # original Kaggle dataset
TARGET_DIR = "dataset"          # output dataset

TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

random.seed(42)

# =========================
# CREATE FOLDERS
# =========================

train_dir = os.path.join(TARGET_DIR, "Training")
test_dir = os.path.join(TARGET_DIR, "Testing")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# =========================
# SPLIT FUNCTION
# =========================

def split_class(class_name):
    src_class_dir = os.path.join(SOURCE_DIR, class_name)

    images = [
        f for f in os.listdir(src_class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)

    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    # create class folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # copy train
    for img in train_imgs:
        shutil.copy2(
            os.path.join(src_class_dir, img),
            os.path.join(train_dir, class_name, img)
        )

    # copy test
    for img in test_imgs:
        shutil.copy2(
            os.path.join(src_class_dir, img),
            os.path.join(test_dir, class_name, img)
        )

    print(f"{class_name}: {len(train_imgs)} train / {len(test_imgs)} test")

# =========================
# RUN FOR ALL CLASSES
# =========================

classes = os.listdir(SOURCE_DIR)

for c in classes:
    split_class(c)

print("\n✅ Dataset split complete!")
print(f"Train dir: {train_dir}")
print(f"Test dir: {test_dir}")
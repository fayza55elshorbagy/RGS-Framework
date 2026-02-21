import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# Step 1: Dataset parameters
image_size = (224, 224)
batch_size = 64

# Step 2: Load images
dataset = tf.keras.utils.image_dataset_from_directory(
    "/content/drive/MyDrive/reduced_kvasir",
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False  # IMPORTANT for label-feature alignment
)

# Step 3: Collect images and labels
X = []
Y = []

for images, labels in dataset:
    X.append(images)
    Y.append(labels)

X = tf.concat(X, axis=0)
Y = tf.concat(Y, axis=0)

print("Dataset shape:", X.shape)

# Step 4: Load ResNet50 (ImageNet pretrained) for feature extraction
print("Loading ResNet50 (ImageNet)...")

model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'   # Global Average Pooling → 2048 features
)

# Step 5: Feature extraction
start_time = time.time()
print("Extracting features...")

X_features = model.predict(X, batch_size=batch_size, verbose=1)

end_time = time.time()

print("Feature extraction time: %.3f s" % (end_time - start_time))
print("Extracted feature dimension:", X_features.shape[1])  # Should be 2048

# Step 6: Save features + labels to CSV
df = pd.DataFrame(X_features)
df['label'] = Y.numpy()

csv_path = "/content/drive/MyDrive/kvasir_resnet50_features.csv"
df.to_csv(csv_path, index=False)

print(f"Features and labels saved at: {csv_path}")
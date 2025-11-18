import sys

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from alexnet import AlexNet

# Configuration
LR = 1e-3
EPOCHS = 5
MODEL_NAME = f'simple-alexnet-epoch{EPOCHS}.model'

# Load command-line arguments
if len(sys.argv) < 3:
    print("Usage: python use_alexnet.py <dataset_path> <labels_path>")
    sys.exit(1)

dataset_path = sys.argv[1]
labels_path = sys.argv[2]

print("[INFO] Loading training data...")
dataset = np.load(dataset_path)
labelset = np.load(labels_path)
WIDTH = dataset.shape[1]
HEIGHT = dataset.shape[2]

# Reshape data to add channel dimension
data = dataset[:, :, :, np.newaxis]

# Split into train and test sets
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labelset, test_size=0.1, random_state=42
)

# Convert labels to one-hot encoding
trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

# Determine number of classes from labels
num_classes = trainLabels.shape[1]

print("[INFO] Compiling model...")
model = AlexNet(width=WIDTH, height=HEIGHT, lr=LR, classes=num_classes)

print("[INFO] Training model...")
model.fit(
    trainData,
    trainLabels,
    batch_size=32,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=1
)

print("[INFO] Evaluating model...")
loss, accuracy = model.evaluate(testData, testLabels, verbose=1)
print(f"[INFO] Accuracy: {accuracy * 100:.2f}%")

# Save the model
print(f"[INFO] Saving model to {MODEL_NAME}...")
model.save(MODEL_NAME)


import sys
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

from lenet import LeNet

# Load command-line arguments
if len(sys.argv) < 3:
    print("Usage: python use_lenet.py <dataset_path> <labels_path>")
    sys.exit(1)

dataset_path = sys.argv[1]
labels_path = sys.argv[2]

print("[INFO] Loading training data...")
dataset = np.load(dataset_path)
labelset = np.load(labels_path)

# Reshape data to add channel dimension
data = dataset[:, :, :, np.newaxis]

# Split into train and test sets
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labelset, test_size=0.2, random_state=42
)

# Convert labels to one-hot encoding
trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

print("[INFO] Compiling model...")
num_classes = len(np.unique(labelset))
opt = SGD(learning_rate=0.01)
model = LeNet(
    width=dataset.shape[1],
    height=dataset.shape[2],
    depth=1,
    classes=num_classes
)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training model...")
model.fit(trainData, trainLabels, batch_size=128, epochs=20, verbose=1)

print("[INFO] Evaluating model...")
loss, accuracy = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print(f"\n[INFO] Accuracy: {accuracy * 100:.2f}%")




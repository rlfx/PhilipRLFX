from typing import Union

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import SGD


def AlexNet(width: int, height: int, lr: float = 1e-3, classes: int = 3) -> Sequential:
    """AlexNet architecture adapted for modern TensorFlow/Keras.

    Note: This is a simplified version adapted for TensorFlow.
    The original AlexNet uses different pooling and normalization strategies.
    Local Response Normalization (LRN) has been replaced with BatchNormalization
    for better performance with modern hardware.

    Args:
        width: Image width
        height: Image height
        lr: Learning rate (default 1e-3)
        classes: Number of output classes (default 3)

    Returns:
        Compiled Sequential model
    """
    model = Sequential([
        Conv2D(96, (11, 11), strides=4, padding='same', activation='relu',
               input_shape=(height, width, 1)),
        MaxPooling2D((3, 3), strides=2, padding='same'),

        Conv2D(256, (5, 5), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=2, padding='same'),

        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D((3, 3), strides=2, padding='same'),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(classes, activation='softmax'),
    ])

    optimizer = SGD(learning_rate=lr, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
        
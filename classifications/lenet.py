from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense


def LeNet(width: int, height: int, depth: int = 1, classes: int = 10) -> Sequential:
    """LeNet CNN architecture for image classification.

    Args:
        width: Image width
        height: Image height
        depth: Image depth/channels (default 1 for grayscale)
        classes: Number of output classes

    Returns:
        Compiled Sequential model
    """
    model = Sequential()

    # CONV => RELU => POOL
    model.add(Conv2D(20, (5, 5), padding="same",
        input_shape=(height, width, depth)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # CONV => RELU => POOL
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model
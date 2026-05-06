from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    LSTM,
    MaxPooling2D,
    TimeDistributed,
)

MAX_LEN = 500
NUM_MFCC = 40
NUM_CLASSES = 5


def build_speech_model(
    max_len: int = MAX_LEN,
    num_mfcc: int = NUM_MFCC,
    num_classes: int = NUM_CLASSES,
) -> Sequential:
    """CNN + Bidirectional LSTM encoder for speech emotion recognition.

    Input shape: (batch, max_len, num_mfcc, 1)
    Output:      (batch, num_classes) softmax probabilities
    """
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same",
                   input_shape=(max_len, num_mfcc, 1)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            TimeDistributed(Flatten()),
            Dropout(0.5),

            Dense(512, activation="relu"),

            Bidirectional(LSTM(units=128)),
            BatchNormalization(),
            Dropout(0.5),

            Dense(num_classes, activation="softmax"),
        ]
    )
    return model

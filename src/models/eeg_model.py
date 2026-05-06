import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Flatten,
    GRU,
    Input,
)

NUM_EEG_CLASSES = 3


def build_eeg_model(
    input_timesteps: int,
    input_features: int,
    num_classes: int = NUM_EEG_CLASSES,
) -> tf.keras.Model:
    """Stacked Bidirectional GRU encoder for EEG emotion recognition.

    Input shape: (batch, input_timesteps, input_features)
    Output:      (batch, num_classes) softmax probabilities
    """
    inputs = Input(shape=(input_timesteps, input_features))

    x = Bidirectional(GRU(64, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(GRU(64, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="eeg_bigru")

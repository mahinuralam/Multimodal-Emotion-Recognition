import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input
from tensorflow.keras.models import Model


def clone_with_prefix(base_model: tf.keras.Model, prefix: str) -> tf.keras.Model:
    """Clone a Keras model and freeze its weights for use as a frozen encoder."""
    cloned = tf.keras.models.clone_model(base_model)
    cloned.set_weights(base_model.get_weights())
    for layer in cloned.layers:
        layer._name = f"{prefix}_{layer.name}"
    cloned.trainable = False
    return cloned


def build_fusion_model(
    speech_model: tf.keras.Model,
    eeg_model: tf.keras.Model,
    num_classes: int,
    fusion_units: int = 256,
    dropout_rate: float = 0.3,
) -> Model:
    """Transformer-fused multimodal model (TMNet).

    Freezes the pre-trained speech and EEG encoders, concatenates their
    output logits, and learns a lightweight fusion classifier.

    Args:
        speech_model: Pre-trained CNN-BiLSTM speech encoder.
        eeg_model:    Pre-trained BiGRU EEG encoder.
        num_classes:  Number of output emotion classes.
        fusion_units: Hidden dimension of the fusion dense layer.
        dropout_rate: Dropout rate applied after the fusion dense layer.

    Returns:
        Compiled Keras Model ready for fusion training.
    """
    frozen_speech = clone_with_prefix(speech_model, "speech")
    frozen_eeg = clone_with_prefix(eeg_model, "eeg")

    speech_input = Input(shape=frozen_speech.input_shape[1:], name="speech_input")
    eeg_input = Input(shape=frozen_eeg.input_shape[1:], name="eeg_input")

    speech_out = frozen_speech(speech_input)
    eeg_out = frozen_eeg(eeg_input)

    combined = Concatenate(name="fusion_concat")([speech_out, eeg_out])
    x = Dense(fusion_units, activation="relu", name="fusion_dense")(combined)
    x = Dropout(dropout_rate, name="fusion_dropout")(x)
    output = Dense(num_classes, activation="softmax", name="fusion_classifier")(x)

    return Model(
        inputs=[speech_input, eeg_input],
        outputs=output,
        name="TMNet_multimodal_fusion",
    )

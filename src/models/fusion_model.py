import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
)
from tensorflow.keras.models import Model

D_MODEL = 128   # projection dimension for each modality token
NUM_HEADS = 4
FF_UNITS = 256  # feed-forward hidden units inside Transformer block
DROPOUT_RATE = 0.1


def clone_with_prefix(base_model: tf.keras.Model, prefix: str) -> tf.keras.Model:
    """Clone a Keras model, rename layers with a prefix, and freeze weights."""
    cloned = tf.keras.models.clone_model(base_model)
    cloned.set_weights(base_model.get_weights())
    for layer in cloned.layers:
        layer._name = f"{prefix}_{layer.name}"
    cloned.trainable = False
    return cloned


def transformer_encoder_block(
    x: tf.Tensor,
    num_heads: int = NUM_HEADS,
    key_dim: int = D_MODEL // NUM_HEADS,
    ff_units: int = FF_UNITS,
    dropout_rate: float = DROPOUT_RATE,
) -> tf.Tensor:
    """Single Transformer encoder block.

    Structure (paper):
        1. Multi-head self-attention — Q, K, V projections;
           attention score = softmax(QKᵀ / √d_k) · V
        2. Add & LayerNorm (residual connection)
        3. Feed-forward network with ReLU activation
        4. Add & LayerNorm (residual connection)
        5. Dropout applied after both sub-layers
    """
    # --- Multi-head self-attention ---
    attn_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name="mha"
    )(x, x)
    attn_output = Dropout(dropout_rate, name="attn_dropout")(attn_output)
    x = LayerNormalization(epsilon=1e-6, name="attn_layernorm")(Add()([x, attn_output]))

    # --- Feed-forward network ---
    ff = Dense(ff_units, activation="relu", name="ffn_dense1")(x)
    ff = Dense(x.shape[-1], name="ffn_dense2")(ff)
    ff = Dropout(dropout_rate, name="ffn_dropout")(ff)
    x = LayerNormalization(epsilon=1e-6, name="ffn_layernorm")(Add()([x, ff]))

    return x


def build_fusion_model(
    speech_model: tf.keras.Model,
    eeg_model: tf.keras.Model,
    num_classes: int,
    d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS,
    ff_units: int = FF_UNITS,
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """TMNet: Transformer-fused multimodal emotion recognition model.

    Architecture (paper, Section 3.3):
        1. Freeze pre-trained speech (CNN-BiLSTM) and EEG (BiGRU) encoders.
        2. Project each encoder's output to d_model — creating one token per modality.
        3. Stack tokens into a 2-token sequence: (batch, 2, d_model).
        4. Apply a Transformer encoder block (MHA + FFN + LayerNorm + residual).
        5. Global average pool over the token sequence.
        6. Dense softmax classification head.

    The two-token design lets the Transformer learn genuine cross-modal attention:
    each head can weight how much the speech representation should attend to the
    EEG representation and vice versa — directly modelling the inter-modal
    dependencies described in the paper.
    """
    frozen_speech = clone_with_prefix(speech_model, "speech")
    frozen_eeg = clone_with_prefix(eeg_model, "eeg")

    speech_input = Input(shape=frozen_speech.input_shape[1:], name="speech_input")
    eeg_input = Input(shape=frozen_eeg.input_shape[1:], name="eeg_input")

    # Encoder forward passes (frozen)
    speech_out = frozen_speech(speech_input)   # (batch, speech_classes)
    eeg_out = frozen_eeg(eeg_input)            # (batch, eeg_classes)

    # Project each modality to d_model and form one token each
    speech_token = Dense(d_model, name="speech_projection")(speech_out)  # (batch, d_model)
    eeg_token = Dense(d_model, name="eeg_projection")(eeg_out)            # (batch, d_model)

    # Stack into a sequence of 2 tokens: (batch, 2, d_model)
    speech_token = Reshape((1, d_model), name="speech_reshape")(speech_token)
    eeg_token = Reshape((1, d_model), name="eeg_reshape")(eeg_token)
    tokens = tf.keras.layers.Concatenate(axis=1, name="token_concat")(
        [speech_token, eeg_token]
    )

    # Transformer encoder block
    tokens = transformer_encoder_block(
        tokens,
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        ff_units=ff_units,
        dropout_rate=dropout_rate,
    )

    # Pool over token dimension → classification head
    x = GlobalAveragePooling1D(name="gap")(tokens)
    x = Dropout(dropout_rate, name="cls_dropout")(x)
    output = Dense(num_classes, activation="softmax", name="classifier")(x)

    return Model(
        inputs=[speech_input, eeg_input],
        outputs=output,
        name="TMNet",
    )

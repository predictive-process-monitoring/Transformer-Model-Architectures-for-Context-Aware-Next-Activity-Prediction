import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_models as tfm


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layer_norm_a = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_b = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_a = layers.Dropout(rate)
        self.dropout_b = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout_a(attn_output, training=training)
        out_a = self.layer_norm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output, training=training)
        return self.layer_norm_b(out_a + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def get_next_activity_model(
    max_case_length,
    context_size,
    vocab_size,
    context_vocab_size,
    output_dim,
    model_type,
    embed_dim=36,
    num_heads=4,
    ff_dim=64,
    without_token_position_embedding: bool = False,
):
    inputs = layers.Input(shape=(max_case_length,))
    context_inputs = []
    if not without_token_position_embedding:
        x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
        x_contexts = []
        for i in range(0, context_size):
            context_inputs.append(layers.Input(shape=(max_case_length,)))
            x_contexts.append(
                TokenAndPositionEmbedding(
                    max_case_length, context_vocab_size, embed_dim
                )(context_inputs[i])
            )
    else:
        x = inputs
        for i in range(0, context_size):
            context_inputs.append(layers.Input(shape=(max_case_length,)))
        x_contexts = context_inputs

    if model_type.startswith("dense"):
        if model_type == "dense-single":
            x = layers.Concatenate()([x] + x_contexts)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Dense(64, activation="relu")(x)
        else:
            x = layers.Dense(64, activation="relu")(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x_contexts = [
                layers.LayerNormalization(epsilon=1e-6)(
                    layers.Dense(64, activation="relu")(x_context)
                )
                for x_context in x_contexts
            ]
            x = layers.Concatenate()([x] + x_contexts)
            x = layers.Dense(64, activation="relu")(x)
    elif model_type.startswith("transformer"):
        embed_multiplier = len([x] + x_contexts)
        if model_type == "transformer-single":
            x = layers.Concatenate()([x] + x_contexts)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = TransformerBlock(embed_dim * embed_multiplier, num_heads, ff_dim)(x)
        else:
            x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
            x_contexts = [
                TransformerBlock(embed_dim, num_heads, ff_dim)(x_context)
                for x_context in x_contexts
            ]
            x = layers.Concatenate(axis=1)([x] + x_contexts)
            # x = x + tfm.nlp.layers.PositionEmbedding(max_length=max_case_length * 2)(x)
            x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
            # x = tfm.nlp.layers.TransformerEncoderBlock(num_heads, embed_dim * embed_multiplier, 'relu')(x)

    elif model_type.startswith("bi-lstm"):
        if model_type == "bi-lstm-single":
            x = layers.Concatenate()([x] + x_contexts)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Bidirectional(
                layers.LSTM(
                    32,
                    implementation=2,
                    return_sequences=True,
                    recurrent_dropout=0.1,
                    dropout=0.1,
                )
            )(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        else:
            x = layers.Bidirectional(
                layers.LSTM(
                    32,
                    implementation=2,
                    return_sequences=True,
                    recurrent_dropout=0.1,
                    dropout=0.1,
                )
            )(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x_contexts = [
                layers.LayerNormalization(epsilon=1e-6)(
                    layers.Bidirectional(
                        layers.LSTM(
                            32,
                            implementation=2,
                            return_sequences=True,
                            recurrent_dropout=0.1,
                            dropout=0.1,
                        )
                    )(x_context)
                )
                for x_context in x_contexts
            ]
            x = layers.Concatenate()([x] + x_contexts)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Bidirectional(
                layers.LSTM(
                    32,
                    implementation=2,
                    return_sequences=True,
                    recurrent_dropout=0.1,
                    dropout=0.1,
                )
            )(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)

    if not without_token_position_embedding:
        x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="linear")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="softmax")(x)
    transformer = tf.keras.Model(
        inputs=[inputs] + context_inputs,
        outputs=outputs,
        name="next_activity_transformer",
    )
    return transformer


def get_next_time_model(
    max_case_length, vocab_size, output_dim=1, embed_dim=36, num_heads=4, ff_dim=64
):

    inputs = layers.Input(shape=(max_case_length,))
    # Three time-based features
    time_inputs = layers.Input(shape=(3,))
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_t = layers.Dense(32, activation="relu")(time_inputs)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(
        inputs=[inputs, time_inputs], outputs=outputs, name="next_time_transformer"
    )
    return transformer


def get_remaining_time_model(
    max_case_length, vocab_size, output_dim=1, embed_dim=36, num_heads=4, ff_dim=64
):

    inputs = layers.Input(shape=(max_case_length,))
    # Three time-based features
    time_inputs = layers.Input(shape=(3,))
    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_t = layers.Dense(32, activation="relu")(time_inputs)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    transformer = tf.keras.Model(
        inputs=[inputs, time_inputs], outputs=outputs, name="remaining_time_transformer"
    )
    return transformer

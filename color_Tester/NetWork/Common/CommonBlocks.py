# from tensorflow.keras import layers, initializers
from keras import layers, initializers


class UnitBlock:
    @staticmethod
    def unit_conv_block(x, filters, kernel_size, strides, kernel_regularizer=None):
        kernel_initializer = initializers.GlorotUniform(seed=1234)
        conv = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False,
                             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
        bn = layers.BatchNormalization(epsilon=0.00001)(conv)
        act = layers.Activation(activation="relu")(bn)
        return act

    @staticmethod
    def unit_conv_transpose_block(x, filters, kernel_size, strides, kernel_regularizer=None):
        kernel_initializer = initializers.GlorotUniform(seed=1234)
        bias_initializer = initializers.Zeros()
        conv = layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding="same", use_bias=True,
                                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                      kernel_regularizer=kernel_regularizer)(x)
        bn = layers.BatchNormalization(epsilon=0.001)(conv)
        act = layers.Activation(activation="relu")(bn)
        return act

    @staticmethod
    def unit_residual_block(x, filters, kernel_size, strides, kernel_regularizer=None):
        kernel_initializer = initializers.GlorotUniform(seed=1234)
        shortcut = x
        conv = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False,
                             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
        bn = layers.BatchNormalization(epsilon=0.00001)(conv)
        act = layers.Activation(activation="relu")(bn)
        conv = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding="same", use_bias=False,
                             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(act)
        bn = layers.BatchNormalization(epsilon=0.00001)(conv)
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides, padding="same", use_bias=False,
                                     kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
            shortcut = layers.BatchNormalization(epsilon=0.00001)(shortcut)
        add = layers.Add()([bn, shortcut])
        act = layers.Activation(activation="relu")(add)
        return act


class EncoderBlock:
    @staticmethod
    def residual_blocks(x, res_block_num, filters, kernel_size, strides, kernel_regularizer=None):
        x = UnitBlock.unit_residual_block(x, filters, kernel_size, strides, kernel_regularizer=kernel_regularizer)
        for _ in range(1, res_block_num):
            x = UnitBlock.unit_residual_block(x, filters, kernel_size, 1, kernel_regularizer=kernel_regularizer)
        return x

    @staticmethod
    def avg_pooling_and_classify(x, units):
        kernel_initializer = initializers.GlorotUniform(seed=1234)
        gap = layers.GlobalAveragePooling2D()(x)
        dense = layers.Dense(units, use_bias=False, kernel_initializer=kernel_initializer)(gap)
        act = layers.Activation(activation="softmax")(dense)
        return act


class UNetDecoderBlock:
    @staticmethod
    def u_net_decoder_block_up_sampling(x, filters, kernel_size, strides, skip=None):
        up_sampling = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
        if skip is not None:
            up_sampling = layers.Concatenate()([up_sampling, skip])
        conv = UnitBlock.unit_conv_block(up_sampling, filters, kernel_size, strides)
        return conv

    @staticmethod
    def u_net_decoder_block_conv_transpose(x, filters, kernel_size, strides, skip=None):
        conv_transpose = UnitBlock.unit_conv_transpose_block(x, filters, 2, 2)
        if skip is not None:
            conv_transpose = layers.Concatenate()([conv_transpose, skip])
        conv = UnitBlock.unit_conv_block(conv_transpose, filters, kernel_size, strides)
        return conv

    @staticmethod
    def u_net_output_block(x, filters, activation, kernel_regularizer=None):
        kernel_initializer = initializers.GlorotUniform(seed=1234)
        conv = layers.Conv2D(filters, kernel_size=1, strides=1, padding="same",
                             kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)
        act = layers.Activation(activation=activation)(conv)
        return act


class TransformerBlock:
    @staticmethod
    def transformer_block(x, num_heads, key_dim, ff_dim):
        # seq_len = x.shape[1]
        # embed_dim = x.shape[2]

        x = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        attn_output = layers.Dropout(0.1)(attn_output)
        out1 = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(out1)
        # ff_dim = embed_dim
        x_ff = layers.Dense(ff_dim, activation="relu")(x)
        # x_ff = layers.Dense(embed_dim)(x_ff)
        x_ff = layers.Dense(out1.shape[-1])(x_ff)
        out2 = layers.Add()([out1, x_ff])
        return out2

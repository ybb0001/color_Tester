import tensorflow as tf
# from tensorflow.python.keras import layers, models, regularizers
from keras import layers, models, regularizers
from NetWork.Common.CommonBlocks import UnitBlock, EncoderBlock, UNetDecoderBlock
from NetWork.Common.Util import Utils


class ResUNet:
    def __init__(self):
        self.model = None
        self.best_model = None
        self.num_class = 0
        self.loss_func = None
        self.metric_func = None
        self.lr_scheduler = None

    def build_model_resunet10(self, input_shape, num_class, decoder_type="up_sampling", l2_reg=None):
        l2_regularizer = None
        if l2_reg is not None:
            l2_regularizer = regularizers.l2(l2_reg)

        # 输入层
        data = layers.Input(shape=input_shape)

        # 编码器（ResNet10 结构）
        x = UnitBlock.unit_conv_block(data, 64, 7, 2,
                                      kernel_regularizer=l2_regularizer)  # 1 个卷积层
        skip_1 = x  # 用于 UNet 跳跃连接
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)  # 1 个最大池化层
        x = EncoderBlock.residual_blocks(x, 2, 64, 3, 1,
                                         kernel_regularizer=l2_regularizer)  # 2*2 个卷积层
        skip_2 = x  # 用于 UNet 跳跃连接
        x = EncoderBlock.residual_blocks(x, 2, 128, 3, 2,
                                         kernel_regularizer=l2_regularizer)  # 2*2 个卷积层（还有 1 个对 shortcut 下采样不计入层数）

        # 解码器
        if decoder_type == "conv_transpose":
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 64, 3, 1, skip_2)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 64, 3, 1, skip_1)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 32, 3, 1)
        else:
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 64, 3, 1, skip_2)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 64, 3, 1, skip_1)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 32, 3, 1)

        # output
        if num_class > 1:
            x = UNetDecoderBlock.u_net_output_block(x, num_class + 1, "softmax")
        else:
            x = UNetDecoderBlock.u_net_output_block(x, 1, "sigmoid")

        inputs = data
        outputs = x

        self.model = models.Model(inputs, outputs)
        self.num_class = num_class

        return

    def build_model_resunet14(self, input_shape, num_class, decoder_type="up_sampling", l2_reg=None):
        l2_regularizer = None
        if l2_reg is not None:
            l2_regularizer = regularizers.l2(l2_reg)

        # 输入层
        data = layers.Input(shape=input_shape)

        # 编码器（ResNet14 结构）
        x = UnitBlock.unit_conv_block(data, 64, 7, 2,
                                      kernel_regularizer=l2_regularizer)  # 1 个卷积层
        skip_1 = x  # 用于 UNet 跳跃连接
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)  # 1 个最大池化层
        x = EncoderBlock.residual_blocks(x, 2, 64, 3, 1,
                                         kernel_regularizer=l2_regularizer)  # 2*2 个卷积层
        skip_2 = x  # 用于 UNet 跳跃连接
        x = EncoderBlock.residual_blocks(x, 2, 128, 3, 2,
                                         kernel_regularizer=l2_regularizer)  # 2*2 个卷积层（还有 1 个对 shortcut 下采样不计入层数）
        skip_3 = x  # 用于 UNet 跳跃连接
        x = EncoderBlock.residual_blocks(x, 2, 256, 3, 2,
                                         kernel_regularizer=l2_regularizer)  # 2*2 个卷积层（还有 1 个对 shortcut 下采样不计入层数）

        # 解码器
        if decoder_type == "conv_transpose":
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 128, 3, 1, skip_3)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 64, 3, 1, skip_2)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 64, 3, 1, skip_1)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 32, 3, 1)
        else:
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 128, 3, 1, skip_3)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 64, 3, 1, skip_2)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 64, 3, 1, skip_1)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 32, 3, 1)

        # output
        if num_class > 1:
            x = UNetDecoderBlock.u_net_output_block(x, num_class + 1, "softmax")
        else:
            x = UNetDecoderBlock.u_net_output_block(x, 1, "sigmoid")

        inputs = data
        outputs = x

        self.model = models.Model(inputs, outputs)
        self.num_class = num_class

        return

    def build_model_resunet18(self, input_shape, num_class, decoder_type="up_sampling", l2_reg=None):
        l2_regularizer = None
        if l2_reg is not None:
            l2_regularizer = regularizers.l2(l2_reg)

        # 输入层
        data = layers.Input(shape=input_shape)

        # 编码器（ResNet18 结构）
        x = UnitBlock.unit_conv_block(data, 64, 7, 2,
                                      kernel_regularizer=l2_regularizer)  # 1 个卷积层
        skip_1 = x  # 用于 UNet 跳跃连接
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)  # 1 个最大池化层
        x = EncoderBlock.residual_blocks(x, 2, 64, 3, 1,
                                         kernel_regularizer=l2_regularizer)  # 2*2 个卷积层
        skip_2 = x  # 用于 UNet 跳跃连接
        x = EncoderBlock.residual_blocks(x, 2, 128, 3, 2,
                                         kernel_regularizer=l2_regularizer)  # 2*2 个卷积层（还有 1 个对 shortcut 下采样不计入层数）
        skip_3 = x  # 用于 UNet 跳跃连接
        x = EncoderBlock.residual_blocks(x, 2, 256, 3, 2,
                                         kernel_regularizer=l2_regularizer)  # 2*2 个卷积层（还有 1 个对 shortcut 下采样不计入层数）
        skip_4 = x  # 用于 UNet 跳跃连接
        x = EncoderBlock.residual_blocks(x, 2, 512, 3, 2,
                                         kernel_regularizer=l2_regularizer)  # 2*2 个卷积层（还有 1 个对 shortcut 下采样不计入层数）

        # 解码器
        if decoder_type == "conv_transpose":
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 256, 3, 1, skip_4)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 128, 3, 1, skip_3)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 64, 3, 1, skip_2)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 64, 3, 1, skip_1)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 32, 3, 1)
        else:
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 256, 3, 1, skip_4)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 128, 3, 1, skip_3)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 64, 3, 1, skip_2)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 64, 3, 1, skip_1)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 32, 3, 1)

        # output
        if num_class > 1:
            x = UNetDecoderBlock.u_net_output_block(x, num_class + 1, "softmax")
        else:
            x = UNetDecoderBlock.u_net_output_block(x, 1, "sigmoid")

        inputs = data
        outputs = x

        self.model = models.Model(inputs, outputs)
        self.num_class = num_class

        return

    def build_model_resunet34(self, input_shape, num_class, decoder_type="up_sampling", l2_reg=None):
        l2_regularizer = None
        if l2_reg is not None:
            l2_regularizer = regularizers.l2(l2_reg)

        # 输入层
        data = layers.Input(shape=input_shape)

        # 编码器（ResNet34 结构）
        x = UnitBlock.unit_conv_block(data, 64, 7, 2,
                                      kernel_regularizer=l2_regularizer)  # 1 个卷积层
        skip_1 = x  # 用于 UNet 跳跃连接
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)  # 1 个最大池化层
        x = EncoderBlock.residual_blocks(x, 3, 64, 3, 1,
                                         kernel_regularizer=l2_regularizer)  # 2*3 个卷积层
        skip_2 = x  # 用于 UNet 跳跃连接
        x = EncoderBlock.residual_blocks(x, 4, 128, 3, 2,
                                         kernel_regularizer=l2_regularizer)  # 2*4 个卷积层（还有 1 个对 shortcut 下采样不计入层数）
        skip_3 = x  # 用于 UNet 跳跃连接
        x = EncoderBlock.residual_blocks(x, 6, 256, 3, 2,
                                         kernel_regularizer=l2_regularizer)  # 2*6 个卷积层（还有 1 个对 shortcut 下采样不计入层数）
        skip_4 = x  # 用于 UNet 跳跃连接
        x = EncoderBlock.residual_blocks(x, 3, 512, 3, 2,
                                         kernel_regularizer=l2_regularizer)  # 2*3 个卷积层（还有 1 个对 shortcut 下采样不计入层数）

        # 解码器
        if decoder_type == "conv_transpose":
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 256, 3, 1, skip_4)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 128, 3, 1, skip_3)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 64, 3, 1, skip_2)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 64, 3, 1, skip_1)
            x = UNetDecoderBlock.u_net_decoder_block_conv_transpose(x, 32, 3, 1)
        else:
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 256, 3, 1, skip_4)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 128, 3, 1, skip_3)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 64, 3, 1, skip_2)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 64, 3, 1, skip_1)
            x = UNetDecoderBlock.u_net_decoder_block_up_sampling(x, 32, 3, 1)

        # output
        if num_class > 1:
            x = UNetDecoderBlock.u_net_output_block(x, num_class + 1, "softmax")
        else:
            x = UNetDecoderBlock.u_net_output_block(x, 1, "sigmoid")

        inputs = data
        outputs = x

        self.model = models.Model(inputs, outputs)
        self.num_class = num_class

        return

    def compile_model(self, loss, metric):
        self.model.compile(optimizer="adam", loss=loss, metrics=[metric])

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler.reset()

    def train_model(self, train_gen, val_gen, steps_per_epoch, validation_steps, epoch, best_model_path, custom_objects,
                    monitor="val_accuracy", mode="auto"):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode=mode
            # save_freq="epoch"
        )
        self.model.fit(train_gen, steps_per_epoch=steps_per_epoch, validation_data=val_gen,
                       validation_steps=validation_steps, epochs=epoch, callbacks=[checkpoint, self.lr_scheduler])
        self.best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)

    def load_best_model(self, best_model_path, custom_objects):
        self.best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)

    def predict_with_best_model(self, inputs):
        predict = self.best_model.predict(inputs)
        return predict

    def export_best_model(self, export_path):
        self.best_model.save(export_path)

    def get_grad_cam_heatmap_with_best_model(self, inputs, predict_idx, activation=None):
        last_conv_name = Utils.find_last_conv_layer(self.best_model)
        grad_model = models.Model([self.best_model.inputs],
                                  [self.best_model.get_layer(last_conv_name).output, self.best_model.output])
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(inputs)
            pred_mask = preds[..., predict_idx]
        grads = tape.gradient(pred_mask, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
        heatmap = tf.squeeze(heatmap, axis=0)

        if activation == "relu":
            heatmap = tf.maximum(heatmap, 0)
            max_heat = tf.math.reduce_max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat
        elif activation == "sigmoid":
            heatmap = 1 / (1 + tf.math.exp(-heatmap))
        elif activation == "tanh":
            heatmap = (1 - tf.math.exp(-2 * heatmap)) / (1 + tf.math.exp(-2 * heatmap))
        else:
            min_heat = tf.math.reduce_min(heatmap)
            heatmap -= min_heat
            max_heat = tf.math.reduce_max(heatmap)
            if max_heat == 0:
                max_heat = 1e-10
            heatmap /= max_heat

        return heatmap

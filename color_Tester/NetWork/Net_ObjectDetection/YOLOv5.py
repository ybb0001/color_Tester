import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from NetWork.Common.CommonBlocks import UnitBlock, EncoderBlock
from NetWork.Common.Util import Utils


def custom_op(input):
    return tf.nn.space_to_depth(input, block_size=2)


class YOLOv5:
    def __init__(self):
        self.model = None
        self.best_model = None
        self.num_class = 0
        self.loss_func = None
        self.metric_func = None
        self.lr_scheduler = None
        self.anchor_info = []
        self.grid_sizes_xy = []

    def build_model(self, input_shape, num_class, l2_reg=None):
        l2_regularizer = None
        if l2_reg is not None:
            l2_regularizer = regularizers.l2(l2_reg)

        # 输入层
        data = layers.Input(shape=input_shape)

        space_to_depth = layers.Lambda(lambda x: tf.nn.space_to_depth(x, block_size=2))(data)
        conv = layers.Conv2D(32, 3, 1, padding="same", use_bias=False)(space_to_depth)
        bn = layers.BatchNormalization()(conv)
        act = layers.Activation("swish")(bn)
        zero_padding = layers.ZeroPadding2D(((1, 0), (1, 0)))(act)
        conv_1 = layers.Conv2D(64, 3, 2, padding="valid", use_bias=False)(zero_padding)
        bn_1 = layers.BatchNormalization()(conv_1)
        act_1 = layers.Activation("swish")(bn_1)
        conv_2 = layers.Conv2D(32, 1, 1, padding="same", use_bias=False)(act_1)
        bn_2 = layers.BatchNormalization()(conv_2)
        act_2 = layers.Activation("swish")(bn_2)
        conv_3 = layers.Conv2D(32, 1, 1, padding="same", use_bias=False)(act_2)
        bn_3 = layers.BatchNormalization()(conv_3)
        act_3 = layers.Activation("swish")(bn_3)
        conv_4 = layers.Conv2D(32, 3, 1, padding="same", use_bias=False)(act_3)
        conv_5 = layers.Conv2D(32, 1, 1, padding="same", use_bias=False)(act_1)
        bn_4 = layers.BatchNormalization()(conv_4)
        bn_5 = layers.BatchNormalization()(conv_5)
        act_4 = layers.Activation("swish")(bn_4)
        act_5 = layers.Activation("swish")(bn_5)
        add = layers.Add()([act_2, act_4])
        concat = layers.Concatenate()([act_5, add])
        conv_6 = layers.Conv2D(64, 1, 1, padding="same", use_bias=False)(concat)
        bn_6 = layers.BatchNormalization()(conv_6)
        act_6 = layers.Activation("swish")(bn_6)
        zero_padding_1 = layers.ZeroPadding2D(((1, 0), (1, 0)))(act_6)
        conv_7 = layers.Conv2D(128, 3, 2, padding="valid", use_bias=False)(zero_padding_1)
        bn_7 = layers.BatchNormalization()(conv_7)
        act_7 = layers.Activation("swish")(bn_7)
        conv_8 = layers.Conv2D(64, 1, 1, padding="same", use_bias=False)(act_7)
        bn_8 = layers.BatchNormalization()(conv_8)
        act_8 = layers.Activation("swish")(bn_8)
        conv_9 = layers.Conv2D(64, 1, 1, padding="same", use_bias=False)(act_8)
        bn_9 = layers.BatchNormalization()(conv_9)
        act_9 = layers.Activation("swish")(bn_9)
        conv_10 = layers.Conv2D(64, 3, 1, padding="same", use_bias=False)(act_9)
        bn_10 = layers.BatchNormalization()(conv_10)
        act_10 = layers.Activation("swish")(bn_10)
        add_1 = layers.Add()([act_8, act_10])
        conv_11 = layers.Conv2D(64, 1, 1, padding="same", use_bias=False)(add_1)
        bn_11 = layers.BatchNormalization()(conv_11)
        act_11 = layers.Activation("swish")(bn_11)
        conv_12 = layers.Conv2D(64, 3, 1, padding="same", use_bias=False)(act_11)
        bn_12 = layers.BatchNormalization()(conv_12)
        act_12 = layers.Activation("swish")(bn_12)
        add_2 = layers.Add()([add_1, act_12])
        conv_13 = layers.Conv2D(64, 1, 1, padding="same", use_bias=False)(add_2)
        bn_13 = layers.BatchNormalization()(conv_13)
        act_13 = layers.Activation("swish")(bn_13)
        conv_14 = layers.Conv2D(64, 3, 1, padding="same", use_bias=False)(act_13)
        conv_15 = layers.Conv2D(64, 1, 1, padding="same", use_bias=False)(act_7)
        bn_14 = layers.BatchNormalization()(conv_14)
        bn_15 = layers.BatchNormalization()(conv_15)
        act_14 = layers.Activation("swish")(bn_14)
        act_15 = layers.Activation("swish")(bn_15)
        add_3 = layers.Add()([add_2, act_14])
        concat_1 = layers.Concatenate()([act_15, add_3])
        conv_16 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(concat_1)
        bn_16 = layers.BatchNormalization()(conv_16)
        act_16 = layers.Activation("swish")(bn_16)
        zero_padding_2 = layers.ZeroPadding2D(((1, 0), (1, 0)))(act_16)
        conv_17 = layers.Conv2D(256, 3, 2, padding="valid", use_bias=False)(zero_padding_2)
        bn_17 = layers.BatchNormalization()(conv_17)
        act_17 = layers.Activation("swish")(bn_17)
        conv_18 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(act_17)
        bn_18 = layers.BatchNormalization()(conv_18)
        act_18 = layers.Activation("swish")(bn_18)
        conv_19 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(act_18)
        bn_19 = layers.BatchNormalization()(conv_19)
        act_19 = layers.Activation("swish")(bn_19)
        conv_20 = layers.Conv2D(128, 3, 1, padding="same", use_bias=False)(act_19)
        bn_20 = layers.BatchNormalization()(conv_20)
        act_20 = layers.Activation("swish")(bn_20)
        add_4 = layers.Add()([act_18, act_20])
        conv_21 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(add_4)
        bn_21 = layers.BatchNormalization()(conv_21)
        act_21 = layers.Activation("swish")(bn_21)
        conv_22 = layers.Conv2D(128, 3, 1, padding="same", use_bias=False)(act_21)
        bn_22 = layers.BatchNormalization()(conv_22)
        act_22 = layers.Activation("swish")(bn_22)
        add_5 = layers.Add()([add_4, act_22])
        conv_23 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(add_5)
        bn_23 = layers.BatchNormalization()(conv_23)
        act_23 = layers.Activation("swish")(bn_23)
        conv_24 = layers.Conv2D(128, 3, 1, padding="same", use_bias=False)(act_23)
        conv_25 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(act_17)
        bn_24 = layers.BatchNormalization()(conv_24)
        bn_25 = layers.BatchNormalization()(conv_25)
        act_24 = layers.Activation("swish")(bn_24)
        act_25 = layers.Activation("swish")(bn_25)
        add_6 = layers.Add()([add_5, act_24])
        concat_2 = layers.Concatenate()([act_25, add_6])
        conv_26 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(concat_2)
        bn_26 = layers.BatchNormalization()(conv_26)
        act_26 = layers.Activation("swish")(bn_26)
        zero_padding_3 = layers.ZeroPadding2D(((1, 0), (1, 0)))(act_26)
        conv_27 = layers.Conv2D(512, 3, 2, padding="valid", use_bias=False)(zero_padding_3)
        bn_27 = layers.BatchNormalization()(conv_27)
        act_27 = layers.Activation("swish")(bn_27)
        conv_28 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(act_27)
        bn_28 = layers.BatchNormalization()(conv_28)
        act_28 = layers.Activation("swish")(bn_28)
        mp = layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding="same")(act_28)
        mp_1 = layers.MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding="same")(act_28)
        mp_2 = layers.MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding="same")(act_28)
        concat_3 = layers.Concatenate()([act_28, mp, mp_1, mp_2])
        conv_29 = layers.Conv2D(512, 1, 1, padding="same", use_bias=False)(concat_3)
        bn_29 = layers.BatchNormalization()(conv_29)
        act_29 = layers.Activation("swish")(bn_29)
        conv_30 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(act_29)
        bn_30 = layers.BatchNormalization()(conv_30)
        act_30 = layers.Activation("swish")(bn_30)
        conv_31 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(act_30)
        bn_31 = layers.BatchNormalization()(conv_31)
        act_31 = layers.Activation("swish")(bn_31)
        conv_32 = layers.Conv2D(256, 3, 1, padding="same", use_bias=False)(act_31)
        conv_33 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(act_29)
        bn_32 = layers.BatchNormalization()(conv_32)
        bn_33 = layers.BatchNormalization()(conv_33)
        act_32 = layers.Activation("swish")(bn_32)
        act_33 = layers.Activation("swish")(bn_33)
        concat_4 = layers.Concatenate()([act_33, act_32])
        conv_34 = layers.Conv2D(512, 1, 1, padding="same", use_bias=False)(concat_4)
        bn_34 = layers.BatchNormalization()(conv_34)
        act_34 = layers.Activation("swish")(bn_34)
        conv_35 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(act_34)
        bn_35 = layers.BatchNormalization()(conv_35)
        act_35 = layers.Activation("swish")(bn_35)
        up_sampling = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(act_35)
        concat_5 = layers.Concatenate()([up_sampling, act_26])
        conv_36 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(concat_5)
        bn_36 = layers.BatchNormalization()(conv_36)
        act_36 = layers.Activation("swish")(bn_36)
        conv_37 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(act_36)
        bn_37 = layers.BatchNormalization()(conv_37)
        act_37 = layers.Activation("swish")(bn_37)
        conv_38 = layers.Conv2D(128, 3, 1, padding="same", use_bias=False)(act_37)
        conv_39 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(concat_5)
        bn_38 = layers.BatchNormalization()(conv_38)
        bn_39 = layers.BatchNormalization()(conv_39)
        act_38 = layers.Activation("swish")(bn_38)
        act_39 = layers.Activation("swish")(bn_39)
        concat_6 = layers.Concatenate()([act_39, act_38])
        conv_40 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(concat_6)
        bn_40 = layers.BatchNormalization()(conv_40)
        act_40 = layers.Activation("swish")(bn_40)
        conv_41 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(act_40)
        bn_41 = layers.BatchNormalization()(conv_41)
        act_41 = layers.Activation("swish")(bn_41)
        up_sampling_1 = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(act_41)
        concat_7 = layers.Concatenate()([up_sampling_1, act_16])

        # CSP2_1
        conv_42 = layers.Conv2D(64, 1, 1, padding="same", use_bias=False)(concat_7)
        bn_42 = layers.BatchNormalization()(conv_42)
        act_42 = layers.Activation("swish")(bn_42)
        conv_43 = layers.Conv2D(64, 1, 1, padding="same", use_bias=False)(act_42)
        bn_43 = layers.BatchNormalization()(conv_43)
        act_43 = layers.Activation("swish")(bn_43)
        conv_44 = layers.Conv2D(64, 3, 1, padding="same", use_bias=False)(act_43)
        conv_45 = layers.Conv2D(64, 1, 1, padding="same", use_bias=False)(concat_7)
        bn_44 = layers.BatchNormalization()(conv_44)
        bn_45 = layers.BatchNormalization()(conv_45)
        act_44 = layers.Activation("swish")(bn_44)
        act_45 = layers.Activation("swish")(bn_45)
        concat_8 = layers.Concatenate()([act_45, act_44])
        conv_46 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(concat_8)
        bn_46 = layers.BatchNormalization()(conv_46)
        act_46 = layers.Activation("swish")(bn_46)

        zero_padding_4 = layers.ZeroPadding2D(((1, 0), (1, 0)))(act_46)
        conv_47 = layers.Conv2D(128, 3, 2, padding="valid", use_bias=False)(zero_padding_4)
        bn_47 = layers.BatchNormalization()(conv_47)
        act_47 = layers.Activation("swish")(bn_47)
        concat_9 = layers.Concatenate()([act_47, act_41])

        # CSP2_1
        conv_48 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(concat_9)
        bn_48 = layers.BatchNormalization()(conv_48)
        act_48 = layers.Activation("swish")(bn_48)
        conv_49 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(act_48)
        bn_49 = layers.BatchNormalization()(conv_49)
        act_49 = layers.Activation("swish")(bn_49)
        conv_50 = layers.Conv2D(128, 3, 1, padding="same", use_bias=False)(act_49)
        conv_51 = layers.Conv2D(128, 1, 1, padding="same", use_bias=False)(concat_9)
        bn_50 = layers.BatchNormalization()(conv_50)
        bn_51 = layers.BatchNormalization()(conv_51)
        act_50 = layers.Activation("swish")(bn_50)
        act_51 = layers.Activation("swish")(bn_51)
        concat_10 = layers.Concatenate()([act_51, act_50])
        conv_52 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(concat_10)
        bn_52 = layers.BatchNormalization()(conv_52)
        act_52 = layers.Activation("swish")(bn_52)

        zero_padding_5 = layers.ZeroPadding2D(((1, 0), (1, 0)))(act_52)
        conv_53 = layers.Conv2D(256, 3, 2, padding="valid", use_bias=False)(zero_padding_5)
        bn_53 = layers.BatchNormalization()(conv_53)
        act_53 = layers.Activation("swish")(bn_53)
        concat_11 = layers.Concatenate()([act_53, act_35])

        # CSP2_1
        conv_54 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(concat_11)
        bn_54 = layers.BatchNormalization()(conv_54)
        act_54 = layers.Activation("swish")(bn_54)
        conv_55 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(act_54)
        bn_55 = layers.BatchNormalization()(conv_55)
        act_55 = layers.Activation("swish")(bn_55)
        conv_56 = layers.Conv2D(256, 3, 1, padding="same", use_bias=False)(act_55)
        conv_57 = layers.Conv2D(256, 1, 1, padding="same", use_bias=False)(concat_11)
        bn_56 = layers.BatchNormalization()(conv_56)
        bn_57 = layers.BatchNormalization()(conv_57)
        act_56 = layers.Activation("swish")(bn_56)
        act_57 = layers.Activation("swish")(bn_57)
        concat_12 = layers.Concatenate()([act_57, act_56])
        conv_58 = layers.Conv2D(512, 1, 1, padding="same", use_bias=False)(concat_12)
        bn_58 = layers.BatchNormalization()(conv_58)
        act_58 = layers.Activation("swish")(bn_58)

        filter_num = 3 * (5 + num_class)
        p5 = layers.Conv2D(filter_num, 1, 1, padding="valid", use_bias=True)(act_58)
        p4 = layers.Conv2D(filter_num, 1, 1, padding="valid", use_bias=True)(act_52)
        p3 = layers.Conv2D(filter_num, 1, 1, padding="valid", use_bias=True)(act_46)

        inputs = [data]
        outputs = [p5, p4, p3]  # 从大物体到小物体
        self.anchor_info = [3, 3, 3]
        self.grid_sizes_xy = [(p5.shape[2], p5.shape[1]), (p4.shape[2], p4.shape[1]), (p3.shape[2], p3.shape[1])]

        self.model = models.Model(inputs, outputs)
        self.num_class = num_class

        return

    def compile_model(self, loss, metric):
        self.model.compile(optimizer="adam", loss=loss, metrics=[metric])

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler.reset()

    def train_model(self, train_gen, val_gen, steps_per_epoch, validation_steps, epoch, best_model_path, custom_objects,
                    monitor="val_loss", mode="auto"):
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
                       validation_steps=validation_steps, epochs=epoch, use_multiprocessing=False, workers=1, callbacks=[checkpoint, self.lr_scheduler])
        self.best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)

    def load_best_model(self, best_model_path, custom_objects):
        self.best_model = tf.keras.models.load_model(best_model_path, custom_objects=custom_objects)

    def predict_with_best_model(self, inputs):
        predict = self.best_model.predict(inputs)
        return predict

    def export_best_model(self, export_path):
        self.best_model.save(export_path)

    def get_anchor_info(self):
        return self.anchor_info

    def get_grid_sizes(self):
        return self.grid_sizes_xy

    def get_grad_cam_heatmap_with_best_model(self, inputs, predict_idx, activation=None):
        last_conv_name = Utils.find_last_conv_layer(self.best_model)
        grad_model = models.Model([self.best_model.inputs],
                                  [self.best_model.get_layer(last_conv_name).output, self.best_model.output])
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(inputs)
            class_channel = preds[:, predict_idx]
        grads = tape.gradient(class_channel, last_conv_layer_output)
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

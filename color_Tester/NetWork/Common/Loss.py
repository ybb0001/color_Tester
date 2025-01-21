import tensorflow as tf


class DynamicWeightedCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, name="dynamic_weighted_cross_entropy"):
        super(DynamicWeightedCrossEntropyLoss, self).__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        positive_weight = tf.reduce_sum(1.0 - y_true) / tf.reduce_sum(y_true)
        weight = tf.where(tf.equal(y_true, 1.0), positive_weight, 1.0)
        if self.from_logits:
            loss = tf.nn.weight_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=weight)
        else:
            loss = -tf.reduce_mean((y_true * tf.math.log(y_pred + 1e-7) * weight + (1.0 - y_true) * tf.math.log(1.0 - y_pred + 1e-7)))

        print(loss.shape)
        return tf.reduce_mean(loss)


class YOLOv5Loss(tf.keras.losses.Loss):
    def __init__(self, img_size, num_classes=20, label_smoothing=0, name="yolo_v5_loss"):
        super(YOLOv5Loss, self).__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        self.num_classes = num_classes
        self.img_width = img_size[0]
        self.img_height = img_size[1]
        self.bce_conf = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.bce_class = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                            label_smoothing=label_smoothing)
        self.anchor_num = 0

    def call(self, y_true, y_pred):
        balance = [0.4, 1.0, 4.0]

        true_box, true_obj, true_class = tf.split(y_true, (4, 1, -1), axis=-1)
        pred_box, pred_obj, pred_class = tf.split(y_pred, (4, 1, -1), axis=-1)
        print(f"true_box.shape{true_box.shape}")
        print(f"true_obj.shape{true_obj.shape}")
        print(f"true_class.shape{true_class.shape}")
        print(f"pred_box.shape{pred_box.shape}")
        print(f"pred_obj.shape{pred_obj.shape}")
        print(f"pred_class.shape{pred_class.shape}")
        if tf.shape(true_class)[-1] == 1 and self.num_classes > 1:
            true_class = tf.squeeze(
                tf.one_hot(tf.cast(true_class, tf.dtypes.int32), depth=self.num_classes, axis=-1), -2)

        # prepare: higher weights to smaller box, true_wh should be normalized to (0,1)
        box_scale = 2 - 1.0 * true_box[..., 2] * true_box[..., 3] / (self.img_width * self.img_height)
        obj_mask = tf.squeeze(true_obj, -1)  # obj or noobj, batch_size * grid * grid * anchors_per_grid
        background_mask = 1.0 - obj_mask
        conf_focal = tf.squeeze(tf.math.pow(true_obj - pred_obj, 2), -1)

        # giou loss
        iou = bbox_iou(pred_box, true_box, xyxy=False)
        iou_loss = (1 - iou) * obj_mask * box_scale  # batch_size * grid * grid * 3

        # confidence loss
        conf_loss = self.bce_conf(true_obj, pred_obj)
        conf_loss = conf_focal * (obj_mask * conf_loss + background_mask * conf_loss)  # batch * grid * grid * 3

        # class loss
        class_loss = obj_mask * self.bce_class(true_class, pred_class)

        print(f"iou_loss.shape{iou_loss.shape}")
        print(f"conf_loss.shape{conf_loss.shape}")
        print(f"class_loss.shape{class_loss.shape}")
        print(f"balance[self.anchor_num]{balance[self.anchor_num]}")
        iou_loss = tf.reduce_mean(iou_loss) * balance[self.anchor_num]
        conf_loss = tf.reduce_mean(conf_loss) * balance[self.anchor_num]
        class_loss = tf.reduce_mean(class_loss) * balance[self.anchor_num]

        self.anchor_num += 1
        if self.anchor_num >= 3:
            self.anchor_num = 0

        return 0.3 * iou_loss + 0.4 * conf_loss + 0.3 * class_loss

        # for i, (pred, true) in enumerate(zip(y_pred, y_true)):
        #     true_box, true_obj, true_class = tf.split(true, (4, 1, -1), axis=-1)
        #     pred_box, pred_obj, pred_class = tf.split(pred, (4, 1, -1), axis=-1)
        #     if tf.shape(true_class)[-1] == 1 and self.num_classes > 1:
        #         true_class = tf.squeeze(
        #             tf.one_hot(tf.cast(true_class, tf.dtypes.int32), depth=self.num_classes, axis=-1), -2)
        #
        #     # prepare: higher weights to smaller box, true_wh should be normalized to (0,1)
        #     box_scale = 2 - 1.0 * true_box[..., 2] * true_box[..., 3] / (self.img_width * self.img_height)
        #     obj_mask = tf.squeeze(true_obj, -1)  # obj or noobj, batch_size * grid * grid * anchors_per_grid
        #     background_mask = 1.0 - obj_mask
        #     conf_focal = tf.squeeze(tf.math.pow(true_obj - pred_obj, 2), -1)
        #
        #     # giou loss
        #     iou = bbox_iou(pred_box, true_box, xyxy=False, giou=True)
        #     iou_loss = (1 - iou) * obj_mask * box_scale  # batch_size * grid * grid * 3
        #
        #     # confidence loss
        #     conf_loss = self.bce_conf(true_obj, pred_obj)
        #     conf_loss = conf_focal * (obj_mask * conf_loss + background_mask * conf_loss)  # batch * grid * grid * 3
        #
        #     # class loss
        #     class_loss = obj_mask * self.bce_class(true_class, pred_class)
        #
        #     iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1, 2, 3]))
        #     conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3]))
        #     class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3]))
        #
        #     iou_loss_all += iou_loss * balance[i]
        #     obj_loss_all += conf_loss * balance[i]
        #     class_loss_all += class_loss * self.num_classes * balance[i]  # to balance the 3 loss
        #
        # return (iou_loss_all, obj_loss_all, class_loss_all)


def bbox_iou(bbox1, bbox2, xyxy=False, epsilon=1e-9):
    # assert bbox1.shape == bbox2.shape
    # giou loss: https://arxiv.org/abs/1902.09630
    if xyxy:
        b1x1, b1y1, b1x2, b1y2 = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2], bbox1[..., 3]
        b2x1, b2y1, b2x2, b2y2 = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2], bbox2[..., 3]
    else:  # xywh -> xyxy
        b1x1, b1x2 = bbox1[..., 0] - bbox1[..., 2] / 2, bbox1[..., 0] + bbox1[..., 2] / 2
        b1y1, b1y2 = bbox1[..., 1] - bbox1[..., 3] / 2, bbox1[..., 1] + bbox1[..., 3] / 2
        b2x1, b2x2 = bbox2[..., 0] - bbox2[..., 2] / 2, bbox2[..., 0] + bbox2[..., 2] / 2
        b2y1, b2y2 = bbox2[..., 1] - bbox2[..., 3] / 2, bbox2[..., 1] + bbox2[..., 3] / 2

    # intersection area
    inter = tf.maximum(tf.minimum(b1x2, b2x2) - tf.maximum(b1x1, b2x1), 0) * \
            tf.maximum(tf.minimum(b1y2, b2y2) - tf.maximum(b1y1, b2y1), 0)

    # union area
    w1, h1 = b1x2 - b1x1 + epsilon, b1y2 - b1y1 + epsilon
    w2, h2 = b2x2 - b2x1 + epsilon, b2y2 - b2y1 + epsilon
    union = w1 * h1 + w2 * h2 - inter + epsilon

    # iou
    iou = inter / union

    cw = tf.maximum(b1x2, b2x2) - tf.minimum(b1x1, b2x1)
    ch = tf.maximum(b1y2, b2y2) - tf.minimum(b1y1, b2y1)
    enclose_area = cw * ch + epsilon
    giou = iou - 1.0 * (enclose_area - union) / enclose_area
    return tf.clip_by_value(giou, -1, 1)


class CustomizedLoss:
    @staticmethod
    def yolo_v5_loss(y_true, y_pred, num_class=2, img_width=1024, img_height=768):
        iou_loss_all = obj_loss_all = class_loss_all = tf.zeros(1)
        true_box, true_obj, true_class = tf.split(y_true, (4, 1, -1), axis=-1)
        pred_box, pred_obj, pred_class = tf.split(y_pred, (4, 1, -1), axis=-1)
        print(f"pred_box.shape{pred_box.shape}")
        if tf.shape(true_class)[-1] == 1:
            true_class = tf.squeeze(
                tf.one_hot(tf.cast(true_class, tf.dtypes.int32), depth=num_class, axis=-1), -2)

            # prepare: higher weights to smaller box, true_wh should be normalized to (0,1)
        box_scale = 2 - 1.0 * true_box[..., 2] * true_box[..., 3] / img_width * img_height
        obj_mask = tf.squeeze(true_obj, -1)  # obj or noobj, batch_size * grid * grid * anchors_per_grid
        background_mask = 1.0 - obj_mask
        conf_focal = tf.squeeze(tf.math.pow(true_obj - pred_obj, 2), -1)

        # giou loss
        iou = bbox_iou(pred_box, true_box, xyxy=False)
        print(f"iou.shape{iou.shape}")
        iou_loss = (1 - iou) * obj_mask * box_scale  # batch_size * grid * grid * 3
        print(f"iou_loss.shape{iou_loss.shape}")

        # confidence loss
        bce_conf = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        conf_loss = bce_conf(true_obj, pred_obj)
        conf_loss = conf_focal * (obj_mask * conf_loss + background_mask * conf_loss)  # batch * grid * grid * 3

        # class loss
        bce_class = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        class_loss = obj_mask * bce_class(true_class, pred_class)

        iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss))
        class_loss = tf.reduce_mean(tf.reduce_sum(class_loss))

        iou_loss_all += iou_loss
        obj_loss_all += conf_loss
        class_loss_all += class_loss * num_class

        return (iou_loss_all, obj_loss_all, class_loss_all)

    @staticmethod
    def weighted_cross_entropy_dice_combined_loss(y_true, y_pred, dice_weight=0.95):
        # 确保 y_pred 是概率分布
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, y_pred.dtype)
        cross_entropy_loss = CustomizedLoss.weighted_cross_entropy_loss(y_true, y_pred, combined=True)
        dice_loss = CustomizedLoss.dice_loss(y_true, y_pred, combined=True)
        return (1.0 - dice_weight) * cross_entropy_loss + dice_weight * dice_loss

    @staticmethod
    def cross_entropy_dice_combined_loss(y_true, y_pred, dice_weight=0.9):
        # 确保 y_pred 是概率分布
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, y_pred.dtype)

        num_classes = y_pred.shape[-1]
        if num_classes == 1:  # 单类别分割
            cross_entropy_loss = tf.losses.binary_crossentropy(y_true, y_pred)
        else:
            cross_entropy_loss = tf.losses.categorical_crossentropy(y_true, y_pred)
        dice_loss = CustomizedLoss.dice_loss(y_true, y_pred, combined=True)
        return (1.0 - dice_weight) * cross_entropy_loss + dice_weight * dice_loss

    @staticmethod
    def weighted_cross_entropy_loss(y_true, y_pred, combined=False):
        # 每对真实值与预测值 动态调节权重

        if not combined:
            # 确保 y_pred 是概率分布
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            y_true = tf.cast(y_true, y_pred.dtype)

        num_classes = y_pred.shape[-1]

        if num_classes == 1:  # 单类别分割
            n_positive = tf.reduce_sum(y_true)
            n_negative = tf.reduce_sum(1.0 - y_true)

            weight_positive = n_negative / (n_positive + tf.keras.backend.epsilon())
            # weight_negative = n_positive / (n_negative + tf.keras.backend.epsilon())
            weight_positive = tf.cast(weight_positive, tf.float32)
            # weight_negative = tf.cast(weight_negative, tf.float32)

            cross_entropy_loss = -(y_true * tf.math.log(y_pred) * weight_positive +
                                   (1 - y_true) * tf.math.log(1 - y_pred))

        else:  # 多类别分割：背景也占一个通道
            total_pixels = tf.reduce_sum(y_true)
            class_pixels = tf.reduce_sum(y_true, axis=[0, 1, 2], keepdims=False)

            weights = (total_pixels - class_pixels) / (class_pixels + tf.keras.backend.epsilon())
            weights = tf.cast(weights, tf.float32)

            cross_entropy_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred) * weights, axis=-1)

        return tf.reduce_mean(cross_entropy_loss)

    @staticmethod
    def cross_entropy_loss(y_true, y_pred, combined=False):
        if not combined:
            # 确保 y_pred 是概率分布
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            y_true = tf.cast(y_true, y_pred.dtype)

        num_classes = y_pred.shape[-1]

        if num_classes == 1:  # 单类别分割
            cross_entropy_loss = -(y_true * tf.math.log(y_pred) +
                                   (1 - y_true) * tf.math.log(1 - y_pred))

        else:  # 多类别分割：背景也占一个通道
            cross_entropy_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        return tf.reduce_mean(cross_entropy_loss)

    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1e-6, combined=False):
        if not combined:
            # 确保 y_pred 是概率分布
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            y_true = tf.cast(y_true, y_pred.dtype)

        num_classes = y_pred.shape[-1]

        if num_classes == 1:  # 单类别分割
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
            dice = (2. * intersection + smooth) / (union + smooth)

        else:  # 多类别分割：背景也占一个通道
            dice = 0.0
            for i in range(num_classes):
                y_true_i = y_true[..., i]
                y_pred_i = y_pred[..., i]

                intersection = tf.reduce_sum(y_true_i * y_pred_i)
                union = tf.reduce_sum(y_true_i) + tf.reduce_sum(y_pred_i)
                dice_i = (2. * intersection + smooth) / (union + smooth)
                dice += dice_i

            dice /= num_classes

        return 1 - dice

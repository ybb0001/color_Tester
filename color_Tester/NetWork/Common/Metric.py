import tensorflow as tf


class CustomizedMetric:
    @staticmethod
    def pixel_accuracy(y_true, y_pred):
        num_classes = tf.shape(y_true)[-1]
        if num_classes == 1:  # 单类别分割
            y_pred_classes = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
            correct_predictions = tf.cast(tf.equal(y_true, y_pred_classes), tf.float32)
            accuracy = tf.reduce_mean(correct_predictions)

        else:  # 多类别分割：背景也占一个通道
            y_true_classes = tf.argmax(y_true, axis=-1)
            y_pred_classes = tf.argmax(y_pred, axis=-1)
            correct_predictions = tf.cast(tf.equal(y_true_classes, y_pred_classes), tf.float32)
            accuracy = tf.reduce_mean(correct_predictions)

        return accuracy

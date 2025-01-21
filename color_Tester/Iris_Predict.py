import os
import glob
import cv2
import numpy as np
import keras as krs
import logging
import tensorflow as tf


filename = "{}.log".format(__file__)
fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

logging.basicConfig(
    level=logging.DEBUG,
    filename=filename,
    filemode="w",
    format=fmt
)



def mLoadModel(mIndex, option):
    logging.debug("Model load Start: %d, %d", mIndex, option)
    cur_folder = os.getcwd()  # 当前工作目录

    class CustomizedLoss:

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

    CUSTOM_OBJECTS = {"weighted_cross_entropy_dice_combined_loss":
                          CustomizedLoss.weighted_cross_entropy_dice_combined_loss,
                      "cross_entropy_dice_combined_loss":
                          CustomizedLoss.cross_entropy_dice_combined_loss,
                      "weighted_cross_entropy_loss":
                          CustomizedLoss.weighted_cross_entropy_loss,
                      "cross_entropy_loss":
                          CustomizedLoss.cross_entropy_loss,
                      "dice_loss":
                          CustomizedLoss.dice_loss,
                      }

    cur_name = "ResUNet10"

    PROJECT_PATH = cur_folder + "\\Data\\Segmentation_Multi"

    all_model_folder = PROJECT_PATH + "\\Models"
    if not os.path.exists(all_model_folder):
        os.makedirs(all_model_folder)
    cur_model_folder = all_model_folder + "\\" + cur_name
    if not os.path.exists(cur_model_folder):
        os.makedirs(cur_model_folder)
    best_model_folder = cur_model_folder + "\\Checkpoint"
    if not os.path.exists(best_model_folder):
        os.makedirs(best_model_folder)
    best_model_path = best_model_folder + "\\best_model"
    logging.debug("load_best_model: %s", best_model_path)
    best_model = krs.models.load_model(best_model_path, custom_objects=CUSTOM_OBJECTS)

    if best_model is None:
        logging.debug("Model load Fail!")
    else:
        logging.debug("Model load Success!")
        logging.debug("best_model: %i", best_model.built)

    return best_model


def mModel_predict( best_model, TEST_IMAGE_PATHS):
    logging.debug("input image Path: %s", TEST_IMAGE_PATHS)
    CLASS_COLORS = [np.array([0, 0, 128], dtype=np.uint8), np.array([0, 128, 0], dtype=np.uint8)]
    cur_folder = os.getcwd()  # 当前工作目录
    PROJECT_PATH = cur_folder + "\\Data\\Segmentation_Multi"
    # TEST_IMAGE_PATHS = sorted(glob.glob(PROJECT_PATH + "\\Images\\Test\\*.bmp"))
    cur_name = "ResUNet10"
    all_model_folder = PROJECT_PATH + "\\Models"
    if not os.path.exists(all_model_folder):
        os.makedirs(all_model_folder)
    cur_model_folder = all_model_folder + "\\" + cur_name
    if not os.path.exists(cur_model_folder):
        os.makedirs(cur_model_folder)
    predict_image_folder = cur_model_folder + "\\Predict"
    if not os.path.exists(predict_image_folder):
        os.makedirs(predict_image_folder)

    logging.debug("Predict Result Ptah: %s", predict_image_folder)

    # for image_path in TEST_IMAGE_PATHS:

    image = cv2.imread(TEST_IMAGE_PATHS)
    image_input = np.expand_dims(image, axis=0).astype(np.float32) / 255.0
    logging.debug("Predict image: %s", TEST_IMAGE_PATHS)
    try:
        model_output = best_model.predict(image_input)
        logging.debug("predict image done")
    except KeyError as e:
        logging.error(repr(e))
        logging.debug("Exception Result: %s", e)
    except Exception as result:
        logging.debug("Exception Result: %s", result)
        logging.error(repr(result))

    if model_output is None:
        logging.error(f"Model output is None for {TEST_IMAGE_PATHS}")
        return 11
    if model_output is None or np.any(np.isnan(model_output)) or np.any(np.isinf(model_output)):
        logging.error(f"Invalid model output for image {TEST_IMAGE_PATHS}")
        return None

    # 转成图像
    logging.debug("Convert image")
    prediction = np.argmax(model_output, axis=-1)
    prediction = prediction[0, :, :]
    height, width = prediction.shape
    num_hole = str(prediction.tolist()).count("1")
    num_blade = str(prediction.tolist()).count("0")
    num_back = str(prediction.tolist()).count("2")

    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(model_output.shape[-1] - 1):
        color_mask[prediction == i] = CLASS_COLORS[i]

    predicted_image = color_mask
    # save image
    base_name = os.path.basename(TEST_IMAGE_PATHS)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(predict_image_folder, f"{name}_predicted{ext}")
    logging.debug("Save image")
    # SegmentationImgProc_Multi.save_predict_image(predicted_image, output_path)
    # cv2.imwrite(output_path, predicted_image)
    if predicted_image is None:
        logging.error(f"Prediction image is None for {TEST_IMAGE_PATHS}")
    else:
        cv2.imwrite(output_path, predicted_image)

    logging.debug("Hole size: %d", num_hole)
    logging.debug("Model Predict done!")
    return num_hole


if __name__ == "__main__":
    MODEL = mLoadModel(10, 1)

    cur_folder = os.getcwd()  # 当前工作目录
    PROJECT_PATH = cur_folder + "\\Data\\Segmentation_Multi"
    TEST_IMAGE_PATHS = sorted(glob.glob(PROJECT_PATH + "\\Images\\Test\\*.bmp"))
    for image_path in TEST_IMAGE_PATHS:
        num_hole=mModel_predict(MODEL, image_path)
        logging.debug("Hole size: %d", num_hole)

import random
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ClassificationImgProc:
    @staticmethod
    def augment_image(image, aug_prob,
                      mirror, flip, rotate90, to_gray,
                      brightness, brightness_delta_min, brightness_delta_max,
                      blur, blur_k_size_max, rotate, angle_min, angle_max,
                      zoom, zoom_out, zoom_in,
                      shift, shift_x, shift_y
                      ):
        if to_gray and (random.random() > aug_prob):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if brightness and (random.random() > aug_prob):
            brightness_delta = random.uniform(brightness_delta_min, brightness_delta_max)
            image = np.clip(image + brightness_delta, 0, 255)
        if blur and (random.random() > aug_prob):
            blur_k_size = random.randint(0, blur_k_size_max)
            if blur_k_size % 2 == 0:
                blur_k_size += 1
            image = cv2.GaussianBlur(np.array(image), (blur_k_size, blur_k_size), 0)

        ori_height, ori_width = image.shape[:2]
        height = ori_height * 2
        width = ori_width * 2
        padding_y = (height - ori_height) // 2
        padding_x = (width - ori_width) // 2
        image = cv2.copyMakeBorder(image, padding_y, height - ori_height - padding_y,
                                   padding_x, width - ori_width - padding_x, cv2.BORDER_CONSTANT, value=0)

        if mirror and (random.random() > aug_prob):
            image = cv2.flip(image, 1)
        if flip and (random.random() > aug_prob):
            image = cv2.flip(image, 0)
        if rotate90 and (random.random() > aug_prob):
            image = cv2.transpose(image)
            direction = random.random()
            if direction > 0.5:
                image = cv2.flip(image, 0)
            else:
                image = cv2.flip(image, 1)
            new_height, new_width = image.shape[:2]
            if new_height >= height:
                top = (new_height - height) // 2
                image = image[top:top + height, :]
            else:
                padding_y = (height - new_height) // 2
                image = cv2.copyMakeBorder(image, padding_y, height - new_height - padding_y,
                                           0, 0, cv2.BORDER_CONSTANT, value=0)
            if new_width >= width:
                left = (new_width - width) // 2
                image = image[:, left:left + width]
            else:
                padding_x = (width - new_width) // 2
                image = cv2.copyMakeBorder(image, 0, 0,
                                           padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
        if rotate and (random.random() > aug_prob):
            angle = random.uniform(angle_min, angle_max)
            m = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
            image = cv2.warpAffine(image, m, (width, height))
        if zoom and (random.random() > aug_prob):
            scale = random.uniform(zoom_out, zoom_in)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
            if new_height >= height:
                top = (new_height - height) // 2
                left = (new_width - width) // 2
                image = image[top:top + height, left:left + width]
            else:
                padding_y = (height - new_height) // 2
                padding_x = (height - new_height) // 2
                image = cv2.copyMakeBorder(image, padding_y, height - new_height - padding_y,
                                           padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
        if shift and (random.random() > aug_prob):
            tx = random.randint(-shift_x, shift_x)
            ty = random.randint(-shift_y, shift_y)
            m = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, m, (width, height))

        top = (height - ori_height) // 2
        left = (width - ori_width) // 2
        image = image[top:top + ori_height, left:left + ori_width]

        return image

    @staticmethod
    def data_generator(images, labels, batch_size, augment=False, aug_prob=0.5,
                       mirror=False, flip=False, rotate90=False, to_gray=False,
                       brightness=False, brightness_delta_min=0.0, brightness_delta_max=0.0,
                       blur=False, blur_k_size_max=1, rotate=False, angle_min=0.0, angle_max=0.0,
                       zoom=False, zoom_out=1.0, zoom_in=1.0,
                       shift=False, shift_x=0, shift_y=0
                       ):
        # cnt = 0
        while True:
            for start in range(0, len(images), batch_size):
                end = min(len(images), start + batch_size)
                batch_images = images[start:end]
                batch_labels = labels[start:end]
                if augment:
                    augmented_images = []
                    for img in batch_images:
                        aug_img = ClassificationImgProc.augment_image(img, aug_prob,
                                                                      mirror, flip, rotate90, to_gray,
                                                                      brightness, brightness_delta_min,
                                                                      brightness_delta_max,
                                                                      blur, blur_k_size_max,
                                                                      rotate, angle_min, angle_max,
                                                                      zoom, zoom_out, zoom_in,
                                                                      shift, shift_x, shift_y
                                                                      )
                        augmented_images.append(aug_img)
                    batch_images = augmented_images
                    # cnt2 = 0
                    # for img in batch_images:
                    #     output_dir = "D:\\wwzx\\Project\\Python\\pythonProject_tensorflow6\\"
                    #     if batch_labels[cnt2][0] == 0:
                    #         output_dir += "aug_ng\\"
                    #         cv2.imwrite(output_dir + str(cnt) + "_img.bmp", img)
                    #         cnt += 1
                    #     else:
                    #         output_dir += "aug_ok\\"
                    #         cv2.imwrite(output_dir + str(cnt) + "_img.bmp", img)
                    #         cnt += 1
                    #     cnt2 += 1
                batch_images = np.array(batch_images, dtype=np.float32) / 255.0
                batch_labels = np.array(batch_labels, dtype=np.float32)

                yield batch_images, batch_labels

    @staticmethod
    def create_train_val_image_list(image_paths, labels, val_size, num_classes):
        images = []

        for img_path in image_paths:
            image = ClassificationImgProc.read_train_val_image(img_path)
            images.append(image)

        labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        labels = labels.tolist()

        train_images, val_images, train_labels, val_labels = train_test_split(images, labels,
                                                                              test_size=val_size,
                                                                              random_state=42)

        print(f"训练集大小：{len(train_images)}")
        print(f"测试集大小：{len(val_images)}")

        return train_images, val_images, train_labels, val_labels

    @staticmethod
    def create_dataset(image_paths, labels, val_size, num_classes):
        images = []

        for img_path in image_paths:
            image = ClassificationImgProc.read_train_val_image(img_path)
            images.append(image)

        images = np.array(images, dtype=np.float32) / 255.0
        labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        labels = labels.tolist()

        train_images, val_images, train_labels, val_labels = train_test_split(images, labels,
                                                                              test_size=val_size,
                                                                              random_state=42)

        print(f"训练集大小：{len(train_images)}")
        print(f"测试集大小：{len(val_images)}")

        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

        return train_dataset, val_dataset

    @staticmethod
    def read_test_image(image_path):
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.imread(image_path)

        return image

    @staticmethod
    def save_predict_image(image, output_path):
        cv2.imwrite(output_path, image)

    @staticmethod
    def read_train_val_image(image_path):
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.imread(image_path)

        return image

    @staticmethod
    def get_label_marked_image(image, mark_color):
        height, width, channel = image.shape
        label_marked = image.copy()
        cur_color = tuple([int(x) for x in mark_color])
        line_thickness = (height + width) // 100
        line_thickness = max(line_thickness, 1)
        cv2.rectangle(label_marked, (0, 0), (width - 1, height - 1), cur_color, line_thickness)

        return label_marked

    @staticmethod
    def get_heatmap_applied_image(image, heatmap):
        height, width, channel = image.shape
        applied_image = image.copy()

        heatmap = cv2.resize(np.array(heatmap), (width, height))
        heatmap = np.uint8(heatmap * 255)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        applied_image = cv2.addWeighted(applied_image, 0.5, heatmap, 0.5, 0)

        return applied_image

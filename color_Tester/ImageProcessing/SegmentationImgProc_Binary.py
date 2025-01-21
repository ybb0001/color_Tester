import random
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SegmentationImgProc_Binary:
    @staticmethod
    def augment_image(image, mask, aug_prob,
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
        mask = cv2.copyMakeBorder(mask, padding_y, height - ori_height - padding_y,
                                  padding_x, width - ori_width - padding_x, cv2.BORDER_CONSTANT, value=0)

        if mirror and (random.random() > aug_prob):
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if flip and (random.random() > aug_prob):
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        if rotate90 and (random.random() > aug_prob):
            image = cv2.transpose(image)
            mask = cv2.transpose(mask)
            direction = random.random()
            if direction > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            else:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            new_height, new_width = image.shape[:2]
            if new_height >= height:
                top = (new_height - height) // 2
                image = image[top:top + height, :]
                mask = mask[top:top + height, :]
            else:
                padding_y = (height - new_height) // 2
                image = cv2.copyMakeBorder(image, padding_y, height - new_height - padding_y,
                                           0, 0, cv2.BORDER_CONSTANT, value=0)
                mask = cv2.copyMakeBorder(mask, padding_y, height - new_height - padding_y,
                                          0, 0, cv2.BORDER_CONSTANT, value=0)
            if new_width >= width:
                left = (new_width - width) // 2
                image = image[:, left:left + width]
                mask = mask[:, left:left + width]
            else:
                padding_x = (width - new_width) // 2
                image = cv2.copyMakeBorder(image, 0, 0,
                                           padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
                mask = cv2.copyMakeBorder(mask, 0, 0,
                                          padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
        if rotate and (random.random() > aug_prob):
            angle = random.uniform(angle_min, angle_max)
            m = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
            image = cv2.warpAffine(image, m, (width, height))
            mask = cv2.warpAffine(mask, m, (width, height))
        if zoom and (random.random() > aug_prob):
            scale = random.uniform(zoom_out, zoom_in)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
            mask = cv2.resize(mask, (new_width, new_height))
            if new_height >= height:
                top = (new_height - height) // 2
                left = (new_width - width) // 2
                image = image[top:top + height, left:left + width]
                mask = mask[top:top + height, left:left + width]
            else:
                padding_y = (height - new_height) // 2
                padding_x = (height - new_height) // 2
                image = cv2.copyMakeBorder(image, padding_y, height - new_height - padding_y,
                                           padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
                mask = cv2.copyMakeBorder(mask, padding_y, height - new_height - padding_y,
                                          padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
        if shift and (random.random() > aug_prob):
            tx = random.randint(-shift_x, shift_x)
            ty = random.randint(-shift_y, shift_y)
            m = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, m, (width, height))
            mask = cv2.warpAffine(mask, m, (width, height))

        top = (height - ori_height) // 2
        left = (width - ori_width) // 2
        image = image[top:top + ori_height, left:left + ori_width]
        mask = mask[top:top + ori_height, left:left + ori_width]

        mask = (mask > 127).astype(np.uint8) * 255
        return image, mask

    @staticmethod
    def data_generator(images, masks, batch_size, augment=False, aug_prob=0.5,
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
                batch_masks = masks[start:end]
                if augment:
                    augmented_images = []
                    augmented_masks = []
                    for img, msk in zip(batch_images, batch_masks):
                        aug_img, aug_msk = SegmentationImgProc_Binary.augment_image(img, msk, aug_prob,
                                                                                    mirror, flip, rotate90, to_gray,
                                                                                    brightness, brightness_delta_min,
                                                                                    brightness_delta_max,
                                                                                    blur, blur_k_size_max,
                                                                                    rotate, angle_min, angle_max,
                                                                                    zoom, zoom_out, zoom_in,
                                                                                    shift, shift_x, shift_y
                                                                                    )
                        augmented_images.append(aug_img)
                        augmented_masks.append(aug_msk)
                    batch_images = augmented_images
                    batch_masks = augmented_masks
                    # for img, msk in zip(batch_images, batch_masks):
                    #     output_dir = "D:\\wwzx\\Project\\Python\\pythonProject_tensorflow6\\aug_biseg\\"
                    #     cv2.imwrite(output_dir + str(cnt) + "_img.bmp", img)
                    #     cv2.imwrite(output_dir + str(cnt) + "_msk.bmp", msk)
                    #     cnt += 1
                batch_images = np.array(batch_images, dtype=np.float32) / 255.0
                batch_masks = np.array(batch_masks, dtype=np.float32) / 255.0
                batch_masks = np.expand_dims(batch_masks, axis=-1)

                yield batch_images, batch_masks

    @staticmethod
    def create_train_val_image_list(image_paths, mask_paths, val_size):
        images = []
        masks = []

        for img_path, msk_path in zip(image_paths, mask_paths):
            image, mask = SegmentationImgProc_Binary.read_train_val_image(img_path, msk_path)
            images.append(image)
            masks.append(mask)

        # weight = compute_class_weights(masks)

        train_images, val_images, train_masks, val_masks = train_test_split(images, masks,
                                                                            test_size=val_size,
                                                                            random_state=42)

        print(f"训练集大小：{len(train_images)}")
        print(f"测试集大小：{len(val_images)}")

        return train_images, val_images, train_masks, val_masks

    @staticmethod
    def create_dataset(image_paths, mask_paths, val_size):
        images = []
        masks = []

        for img_path, msk_path in zip(image_paths, mask_paths):
            image, mask = SegmentationImgProc_Binary.read_train_val_image(img_path, msk_path)
            images.append(image)
            masks.append(mask)

        images = np.array(images, dtype=np.float32) / 255.0
        masks = np.array(masks, dtype=np.float32) / 255.0
        masks = np.expand_dims(masks, axis=-1)

        train_images, val_images, train_masks, val_masks = train_test_split(images, masks,
                                                                            test_size=val_size,
                                                                            random_state=42)

        print(f"训练集大小：{len(train_images)}")
        print(f"测试集大小：{len(val_images)}")

        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))

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
    def read_train_val_image(image_path, mask_path):
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        return image, mask

    @staticmethod
    def get_heatmap_applied_image(image, heatmap):
        height, width, channel = image.shape
        applied_image = image.copy()

        heatmap = cv2.resize(np.array(heatmap), (width, height))
        heatmap = np.uint8(heatmap * 255)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        applied_image = cv2.addWeighted(applied_image, 0.5, heatmap, 0.5, 0)

        return applied_image

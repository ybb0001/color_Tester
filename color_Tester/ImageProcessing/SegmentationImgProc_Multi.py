import random
import copy
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SegmentationImgProc_Multi:
    @staticmethod
    def augment_image(image, origin_mask_list, aug_prob,
                      mirror, flip, rotate90, to_gray,
                      brightness, brightness_delta_min, brightness_delta_max,
                      blur, blur_k_size_max, rotate, angle_min, angle_max,
                      zoom, zoom_out, zoom_in,
                      shift, shift_x, shift_y
                      ):
        mask_list = copy.deepcopy(origin_mask_list)  # 深拷贝...
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
        for i in range(len(mask_list) - 1):
            mask_list[i] = cv2.copyMakeBorder(mask_list[i], padding_y, height - ori_height - padding_y,
                                              padding_x, width - ori_width - padding_x, cv2.BORDER_CONSTANT, value=0)

        if mirror and (random.random() > aug_prob):
            image = cv2.flip(image, 1)
            for i in range(len(mask_list) - 1):
                mask_list[i] = cv2.flip(mask_list[i], 1)
        if flip and (random.random() > aug_prob):
            image = cv2.flip(image, 0)
            for i in range(len(mask_list) - 1):
                mask_list[i] = cv2.flip(mask_list[i], 0)
        if rotate90 and (random.random() > aug_prob):
            image = cv2.transpose(image)
            for i in range(len(mask_list) - 1):
                mask_list[i] = cv2.transpose(mask_list[i])
            direction = random.random()
            if direction > 0.5:
                image = cv2.flip(image, 0)
                for i in range(len(mask_list) - 1):
                    mask_list[i] = cv2.flip(mask_list[i], 0)
            else:
                image = cv2.flip(image, 1)
                for i in range(len(mask_list) - 1):
                    mask_list[i] = cv2.flip(mask_list[i], 1)
            new_height, new_width = image.shape[:2]
            if new_height >= height:
                top = (new_height - height) // 2
                image = image[top:top + height, :]
                for i in range(len(mask_list) - 1):
                    mask_list[i] = mask_list[i][top:top + height, :]
            else:
                padding_y = (height - new_height) // 2
                image = cv2.copyMakeBorder(image, padding_y, height - new_height - padding_y,
                                           0, 0, cv2.BORDER_CONSTANT, value=0)
                for i in range(len(mask_list) - 1):
                    mask_list[i] = cv2.copyMakeBorder(mask_list[i], padding_y, height - new_height - padding_y,
                                                      0, 0, cv2.BORDER_CONSTANT, value=0)
            if new_width >= width:
                left = (new_width - width) // 2
                image = image[:, left:left + width]
                for i in range(len(mask_list) - 1):
                    mask_list[i] = mask_list[i][:, left:left + width]
            else:
                padding_x = (width - new_width) // 2
                image = cv2.copyMakeBorder(image, 0, 0,
                                           padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
                for i in range(len(mask_list) - 1):
                    mask_list[i] = cv2.copyMakeBorder(mask_list[i], 0, 0,
                                                      padding_x, width - new_width - padding_x,
                                                      cv2.BORDER_CONSTANT, value=0)
        if rotate and (random.random() > aug_prob):
            angle = random.uniform(angle_min, angle_max)
            m = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
            image = cv2.warpAffine(image, m, (width, height))
            for i in range(len(mask_list) - 1):
                mask_list[i] = cv2.warpAffine(mask_list[i], m, (width, height))
        if zoom and (random.random() > aug_prob):
            scale = random.uniform(zoom_out, zoom_in)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
            for i in range(len(mask_list) - 1):
                mask_list[i] = cv2.resize(mask_list[i], (new_width, new_height))
            if new_height >= height:
                top = (new_height - height) // 2
                left = (new_width - width) // 2
                image = image[top:top + height, left:left + width]
                for i in range(len(mask_list) - 1):
                    mask_list[i] = mask_list[i][top:top + height, left:left + width]
            else:
                padding_y = (height - new_height) // 2
                padding_x = (height - new_height) // 2
                image = cv2.copyMakeBorder(image, padding_y, height - new_height - padding_y,
                                           padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
                for i in range(len(mask_list) - 1):
                    mask_list[i] = cv2.copyMakeBorder(mask_list[i], padding_y, height - new_height - padding_y,
                                                      padding_x, width - new_width - padding_x,
                                                      cv2.BORDER_CONSTANT, value=0)
        if shift and (random.random() > aug_prob):
            tx = random.randint(-shift_x, shift_x)
            ty = random.randint(-shift_y, shift_y)
            m = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, m, (width, height))
            for i in range(len(mask_list) - 1):
                mask_list[i] = cv2.warpAffine(mask_list[i], m, (width, height))

        top = (height - ori_height) // 2
        left = (width - ori_width) // 2
        image = image[top:top + ori_height, left:left + ori_width]
        for i in range(len(mask_list) - 1):
            mask_list[i] = mask_list[i][top:top + ori_height, left:left + ori_width]

        for i in range(len(mask_list) - 1):
            mask_list[i] = (mask_list[i] > 127).astype(np.uint8) * 255
        intersection = mask_list[0]
        for i in range(1, len(mask_list) - 1, 1):
            intersection = cv2.bitwise_or(intersection, mask_list[i])
        mask_list[len(mask_list) - 1] = cv2.bitwise_not(intersection)
        return image, mask_list

    @staticmethod
    def data_generator(images, mask_lists, batch_size, augment=False, aug_prob=0.5,
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
                batch_mask_lists = mask_lists[start:end]
                if augment:
                    augmented_images = []
                    augmented_mask_lists = []
                    for img, msk_list in zip(batch_images, batch_mask_lists):
                        aug_img, aug_msk_list = SegmentationImgProc_Multi.augment_image(img, msk_list, aug_prob,
                                                                                        mirror, flip, rotate90, to_gray,
                                                                                        brightness, brightness_delta_min,
                                                                                        brightness_delta_max,
                                                                                        blur, blur_k_size_max,
                                                                                        rotate, angle_min, angle_max,
                                                                                        zoom, zoom_out, zoom_in,
                                                                                        shift, shift_x, shift_y
                                                                                        )
                        augmented_images.append(aug_img)
                        augmented_mask_lists.append(aug_msk_list)

                    batch_images = augmented_images
                    batch_mask_lists = augmented_mask_lists
                    # for img, msk_list in zip(batch_images, batch_mask_lists):
                    #     cnt2 = 0
                    #     output_dir = "D:\\wwzx\\Project\\Python\\pythonProject_tensorflow6\\aug_mulseg\\"
                    #     cv2.imwrite(output_dir + str(cnt) + "_img.bmp", img)
                    #     for msk in msk_list:
                    #         cv2.imwrite(output_dir + str(cnt) + "_msk_" + str(cnt2) + ".bmp", msk)
                    #         cnt2 += 1
                    #     cnt += 1
                batch_masks = []
                for msk_list in batch_mask_lists:
                    msk = np.stack(msk_list, axis=-1)
                    batch_masks.append(msk)
                batch_images = np.array(batch_images, dtype=np.float32) / 255.0
                batch_masks = np.array(batch_masks, dtype=np.float32) / 255.0

                yield batch_images, batch_masks

    @staticmethod
    def create_train_val_image_list(image_paths, mask_paths, val_size, class_colors):
        images = []
        mask_lists = []

        for img_path, msk_path in zip(image_paths, mask_paths):
            image, mask = SegmentationImgProc_Multi.read_train_val_image(img_path, msk_path)
            images.append(image)
            mask_list = SegmentationImgProc_Multi.separate_each_class_mask(mask, class_colors)
            mask_lists.append(mask_list)

        # weight = compute_class_weights(masks)

        train_images, val_images, train_mask_lists, val_mask_lists = train_test_split(images, mask_lists,
                                                                                      test_size=val_size,
                                                                                      random_state=42)

        print(f"训练集大小：{len(train_images)}")
        print(f"测试集大小：{len(val_images)}")

        return train_images, val_images, train_mask_lists, val_mask_lists

    @staticmethod
    def create_dataset(image_paths, mask_paths, val_size, class_colors):
        images = []
        mask_lists = []

        for img_path, msk_path in zip(image_paths, mask_paths):
            image, mask = SegmentationImgProc_Multi.read_train_val_image(img_path, msk_path)
            images.append(image)
            mask_list = SegmentationImgProc_Multi.separate_each_class_mask(mask, class_colors)
            mask_lists.append(mask_list)

        masks = []
        for msk_list in mask_lists:
            msk = np.stack(msk_list, axis=-1)
            masks.append(msk)

        images = np.array(images, dtype=np.float32) / 255.0
        masks = np.array(masks, dtype=np.float32) / 255.0

        train_images, val_images, train_masks, val_masks = train_test_split(images, masks,
                                                                            test_size=val_size,
                                                                            random_state=42)

        print(f"训练集大小：{len(train_images)}")
        print(f"测试集大小：{len(val_images)}")

        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))

        return train_dataset, val_dataset

    @staticmethod
    def separate_each_class_mask(mask, class_colors):
        mask_list = []
        for i in range(len(class_colors)):
            mask_separated = cv2.inRange(mask, class_colors[i], class_colors[i])
            mask_list.append(mask_separated)

        back_ground_color = np.array([0, 0, 0], dtype=np.uint8)
        mask_separated = cv2.inRange(mask, back_ground_color, back_ground_color)
        mask_list.append(mask_separated)

        return mask_list

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
        mask = cv2.imread(mask_path)

        return image, mask

    @staticmethod
    def generate_predict_image(model_output, class_colors):
        prediction = np.argmax(model_output, axis=-1)
        prediction = prediction[0, :, :]
        height, width = prediction.shape
        color_mask = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(model_output.shape[-1] - 1):
            color_mask[prediction == i] = class_colors[i]

        return color_mask

    @staticmethod
    def get_heatmap_applied_image(image, heatmap):
        height, width, channel = image.shape
        applied_image = image.copy()

        heatmap = cv2.resize(np.array(heatmap), (width, height))
        heatmap = np.uint8(heatmap * 255)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        applied_image = cv2.addWeighted(applied_image, 0.5, heatmap, 0.5, 0)

        return applied_image

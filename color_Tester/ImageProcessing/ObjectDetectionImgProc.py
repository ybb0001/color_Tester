import random
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt


class ObjectDetectionImgProc:
    @staticmethod
    def augment_image(image, boxes, aug_prob,
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
        for i in range(len(boxes)):
            boxes[i][0] = boxes[i][0] + padding_x
            boxes[i][1] = boxes[i][1] + padding_y

        if mirror and (random.random() > aug_prob):
            image = cv2.flip(image, 1)
            for i in range(len(boxes)):
                boxes[i][0] = width - 1 - boxes[i][0] - boxes[i][2]
        if flip and (random.random() > aug_prob):
            image = cv2.flip(image, 0)
            for i in range(len(boxes)):
                boxes[i][1] = height - 1 - boxes[i][1] - boxes[i][3]
        if rotate90 and (random.random() > aug_prob):
            image = cv2.transpose(image)
            for i in range(len(boxes)):
                ori_box_x = boxes[i][0]
                ori_box_y = boxes[i][1]
                ori_box_w = boxes[i][2]
                ori_box_h = boxes[i][3]
                boxes[i][0] = ori_box_y
                boxes[i][1] = ori_box_x
                boxes[i][2] = ori_box_h
                boxes[i][3] = ori_box_w
            direction = random.random()
            if direction > 0.5:
                image = cv2.flip(image, 0)
                for i in range(len(boxes)):
                    boxes[i][1] = height - 1 - boxes[i][1] - boxes[i][3]
            else:
                image = cv2.flip(image, 1)
                for i in range(len(boxes)):
                    boxes[i][0] = width - 1 - boxes[i][0] - boxes[i][2]
            new_height, new_width = image.shape[:2]
            if new_height >= height:
                top = (new_height - height) // 2
                image = image[top:top + height, :]
                for i in range(len(boxes)):
                    boxes[i][1] = boxes[i][1] - top
            else:
                padding_y = (height - new_height) // 2
                image = cv2.copyMakeBorder(image, padding_y, height - new_height - padding_y,
                                           0, 0, cv2.BORDER_CONSTANT, value=0)
                for i in range(len(boxes)):
                    boxes[i][1] = boxes[i][1] + padding_y
            if new_width >= width:
                left = (new_width - width) // 2
                image = image[:, left:left + width]
                for i in range(len(boxes)):
                    boxes[i][0] = boxes[i][0] - left
            else:
                padding_x = (width - new_width) // 2
                image = cv2.copyMakeBorder(image, 0, 0,
                                           padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
                for i in range(len(boxes)):
                    boxes[i][0] = boxes[i][0] + padding_x
        if rotate and (random.random() > aug_prob):
            angle = random.uniform(angle_min, angle_max)
            m = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
            image = cv2.warpAffine(image, m, (width, height))
            for i in range(len(boxes)):
                ori_box_lt = np.array([boxes[i][0], boxes[i][1], 1],
                                      dtype=np.float32).reshape((3, 1))
                ori_box_rt = np.array([boxes[i][0] + boxes[i][2], boxes[i][1], 1],
                                      dtype=np.float32).reshape((3, 1))
                ori_box_lb = np.array([boxes[i][0], boxes[i][1] + boxes[i][3], 1],
                                      dtype=np.float32).reshape((3, 1))
                ori_box_rb = np.array([boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3], 1],
                                      dtype=np.float32).reshape((3, 1))
                warp_box_lt = np.matmul(m, ori_box_lt)
                warp_box_rt = np.matmul(m, ori_box_rt)
                warp_box_lb = np.matmul(m, ori_box_lb)
                warp_box_rb = np.matmul(m, ori_box_rb)
                # 计算最小外接？？？
                boxes[i][0] = min(warp_box_lt[0], warp_box_rt[0], warp_box_lb[0], warp_box_rb[0])
                boxes[i][1] = min(warp_box_lt[1], warp_box_rt[1], warp_box_lb[1], warp_box_rb[1])
                boxes[i][2] = max(warp_box_lt[0], warp_box_rt[0], warp_box_lb[0], warp_box_rb[0]) - boxes[i][0]
                boxes[i][3] = max(warp_box_lt[1], warp_box_rt[1], warp_box_lb[1], warp_box_rb[1]) - boxes[i][1]

        if zoom and (random.random() > aug_prob):
            scale = random.uniform(zoom_out, zoom_in)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
            for i in range(len(boxes)):
                boxes[i][0] = boxes[i][0] * scale
                boxes[i][1] = boxes[i][1] * scale
                boxes[i][2] = boxes[i][2] * scale
                boxes[i][3] = boxes[i][3] * scale
            if new_height >= height:
                top = (new_height - height) // 2
                left = (new_width - width) // 2
                image = image[top:top + height, left:left + width]
                for i in range(len(boxes)):
                    boxes[i][0] = boxes[i][0] - left
                    boxes[i][1] = boxes[i][1] - top
            else:
                padding_y = (height - new_height) // 2
                padding_x = (height - new_height) // 2
                image = cv2.copyMakeBorder(image, padding_y, height - new_height - padding_y,
                                           padding_x, width - new_width - padding_x, cv2.BORDER_CONSTANT, value=0)
                for i in range(len(boxes)):
                    boxes[i][0] = boxes[i][0] + padding_x
                    boxes[i][1] = boxes[i][1] + padding_y
        if shift and (random.random() > aug_prob):
            tx = random.randint(-shift_x, shift_x)
            ty = random.randint(-shift_y, shift_y)
            m = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, m, (width, height))
            for i in range(len(boxes)):
                ori_box_lt = np.array([boxes[i][0], boxes[i][1], 1],
                                      dtype=np.float32).reshape((3, 1))
                ori_box_rt = np.array([boxes[i][0] + boxes[i][2], boxes[i][1], 1],
                                      dtype=np.float32).reshape((3, 1))
                ori_box_lb = np.array([boxes[i][0], boxes[i][1] + boxes[i][3], 1],
                                      dtype=np.float32).reshape((3, 1))
                ori_box_rb = np.array([boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3], 1],
                                      dtype=np.float32).reshape((3, 1))
                warp_box_lt = np.matmul(m, ori_box_lt)
                warp_box_rt = np.matmul(m, ori_box_rt)
                warp_box_lb = np.matmul(m, ori_box_lb)
                warp_box_rb = np.matmul(m, ori_box_rb)
                # 计算最小外接？？？
                boxes[i][0] = min(warp_box_lt[0], warp_box_rt[0], warp_box_lb[0], warp_box_rb[0])
                boxes[i][1] = min(warp_box_lt[1], warp_box_rt[1], warp_box_lb[1], warp_box_rb[1])
                boxes[i][2] = max(warp_box_lt[0], warp_box_rt[0], warp_box_lb[0], warp_box_rb[0]) - boxes[i][0]
                boxes[i][3] = max(warp_box_lt[1], warp_box_rt[1], warp_box_lb[1], warp_box_rb[1]) - boxes[i][1]

        top = (height - ori_height) // 2
        left = (width - ori_width) // 2
        image = image[top:top + ori_height, left:left + ori_width]
        for i in range(len(boxes)):
            boxes[i][0] = boxes[i][0] - left
            boxes[i][1] = boxes[i][1] - top

        aug_boxes = []
        for box in boxes:
            if ((box[0] >= 0) and (box[1] >= 0)
                    and (box[0] + box[2] <= ori_width - 1) and (box[1] + box[3] <= ori_height - 1)):
                aug_boxes.append(box)
        aug_boxes = np.array(aug_boxes, dtype=np.float32)

        return image, aug_boxes

    @staticmethod
    def data_generator(images, boxes_list, anchors, image_size_xy, grid_sizes_xy_list, num_class, batch_size,
                       augment=False, aug_prob=0.5,
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
                batch_boxes_list = boxes_list[start:end]
                if augment:
                    augmented_images = []
                    augmented_boxes_list = []
                    for img, boxes in zip(batch_images, batch_boxes_list):
                        aug_img, aug_boxes = ObjectDetectionImgProc.augment_image(img, boxes, aug_prob,
                                                                                  mirror, flip, rotate90, to_gray,
                                                                                  brightness, brightness_delta_min,
                                                                                  brightness_delta_max,
                                                                                  blur, blur_k_size_max,
                                                                                  rotate, angle_min, angle_max,
                                                                                  zoom, zoom_out, zoom_in,
                                                                                  shift, shift_x, shift_y
                                                                                  )
                        augmented_images.append(aug_img)
                        augmented_boxes_list.append(aug_boxes)
                    batch_images = augmented_images
                    batch_boxes_list = augmented_boxes_list
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
                batch_y_trues = [[] for _ in range(len(anchors))]
                for boxes in batch_boxes_list:
                    for i in range(len(anchors)):
                        y_true = ObjectDetectionImgProc.generate_y_true(boxes, anchors[i],
                                                                        image_size_xy, grid_sizes_xy_list[i], num_class)
                        batch_y_trues[i].append(y_true)
                for i in range(len(anchors)):
                    batch_y_trues[i] = np.array(batch_y_trues[i], dtype=np.float32)

                yield batch_images, batch_y_trues

    @staticmethod
    def create_train_val_image_list(image_paths, bounding_boxes_list, val_size):
        images = []

        for img_path in image_paths:
            image = ObjectDetectionImgProc.read_train_val_image(img_path)
            images.append(image)

        train_images, val_images, train_boxes_list, val_boxes_list = train_test_split(images,
                                                                                      bounding_boxes_list,
                                                                                      test_size=val_size,
                                                                                      random_state=42)

        print(f"训练集大小：{len(train_images)}")
        print(f"测试集大小：{len(val_images)}")

        return train_images, val_images, train_boxes_list, val_boxes_list

    @staticmethod
    def create_dataset(image_paths, bounding_boxes_list, anchors, image_size_xy, grid_sizes_xy_list, num_class):
        images = []
        y_true = [[], [], []]

        for img_path, boxes_list in zip(image_paths, bounding_boxes_list):
            image = ObjectDetectionImgProc.read_train_val_image(img_path)
            images.append(image)
            for i in range(len(anchors)):
                aaa = ObjectDetectionImgProc.generate_y_true(boxes_list, anchors[i],
                                                             image_size_xy, grid_sizes_xy_list[i], num_class)
                y_true[i].append(aaa)

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
    def get_anchors_k_means(bounding_boxes_list, anchor_num_info):
        boxes_size = []

        for boxes in bounding_boxes_list:
            for box in boxes:
                boxes_size.append((box[2], box[3]))

        boxes_size = np.array(boxes_size, dtype=np.float32)

        num_clusters = 0
        for num in anchor_num_info:
            num_clusters += num

        k_means = KMeans(n_clusters=num_clusters, random_state=0).fit(boxes_size)
        k_means_anchors = k_means.cluster_centers_

        k_means_anchors = k_means_anchors[np.argsort(k_means_anchors[:, 0] * k_means_anchors[:, 1])[::-1]]

        anchors = []
        start = 0
        end = 0
        for i in range(len(anchor_num_info)):
            end += anchor_num_info[i]
            cur_anchors = k_means_anchors[start:end]
            anchors.append(cur_anchors)
            print(f"Anchors for output_{i}：\n", cur_anchors)
            start += anchor_num_info[i]

        return anchors

    @staticmethod
    def calc_iou(box, anchor_size, anchor_center):
        box_lt_x = box[0]
        box_lt_y = box[1]
        box_rb_x = box[0] + box[2]
        box_rb_y = box[1] + box[3]
        anchor_lt_x = anchor_center[0] - anchor_size[0] / 2
        anchor_lt_y = anchor_center[1] - anchor_size[1] / 2
        anchor_rb_x = anchor_center[0] + anchor_size[0] / 2
        anchor_rb_y = anchor_center[1] + anchor_size[1] / 2

        inter_lt_x = max(box_lt_x, anchor_lt_x)
        inter_lt_y = max(box_lt_y, anchor_lt_y)
        inter_rb_x = min(box_rb_x, anchor_rb_x)
        inter_rb_y = min(box_rb_y, anchor_rb_y)

        inter_area = max(inter_rb_x - inter_lt_x, 0) * max(inter_rb_y - inter_lt_y, 0)
        box_area = box[2] * box[3]
        anchor_area = anchor_size[0] * anchor_size[1]
        union_area = box_area + anchor_area - inter_area

        iou_value = inter_area / (union_area + 1e-6)

        return iou_value

    @staticmethod
    def generate_y_true(boxes, anchor, image_size_xy, grid_size_xy, num_class):
        y_true = np.zeros((grid_size_xy[1], grid_size_xy[0], anchor.shape[0] * (5 + num_class)), dtype=np.float32)

        grid_unit_x = image_size_xy[0] // grid_size_xy[0]
        grid_unit_y = image_size_xy[1] // grid_size_xy[1]
        box_assign = [[[] for _ in range(grid_size_xy[0])] for _ in range(grid_size_xy[1])]
        for box in boxes:
            box_cen_x = (box[0] + box[0] + box[2]) / 2
            box_cen_y = (box[1] + box[1] + box[3]) / 2
            box_grid_idx_x = int(box_cen_x // grid_unit_x)
            box_grid_idx_y = int(box_cen_y // grid_unit_y)
            box_assign[box_grid_idx_y][box_grid_idx_x].append(box)

        for y in range(grid_size_xy[1]):
            for x in range(grid_size_xy[0]):
                if len(box_assign[y][x]) != 0:
                    iou_matrix = np.zeros((len(anchor), len(box_assign[y][x])), dtype=np.float32)
                    for box_idx in range(len(box_assign[y][x])):
                        for anc_idx in range(len(anchor)):
                            anc_center = (grid_unit_x * x + grid_unit_x // 2, grid_unit_y * y + grid_unit_y // 2)
                            iou_matrix[anc_idx][box_idx] = ObjectDetectionImgProc.calc_iou(
                                box_assign[y][x][box_idx], anchor[anc_idx], anc_center)
                    acn_assigned = 0
                    for _ in range(len(box_assign[y][x])):
                        max_idx = np.argmax(iou_matrix)
                        max_idx_anc, max_idx_box = np.unravel_index(max_idx, iou_matrix.shape)
                        cur_box = box_assign[y][x][max_idx_box]
                        shift_x = ((cur_box[0] + cur_box[0] + cur_box[2]) / 2 -
                                   (grid_unit_x * x + grid_unit_x // 2))
                        shift_y = ((cur_box[1] + cur_box[1] + cur_box[3]) / 2 -
                                   (grid_unit_y * y + grid_unit_y // 2))
                        width_scale = cur_box[2] / anchor[max_idx_anc][0]
                        height_scale = cur_box[3] / anchor[max_idx_anc][1]
                        y_true[y][x][max_idx_anc * (5 + num_class) + 0] = shift_x
                        y_true[y][x][max_idx_anc * (5 + num_class) + 1] = shift_y
                        y_true[y][x][max_idx_anc * (5 + num_class) + 2] = width_scale
                        y_true[y][x][max_idx_anc * (5 + num_class) + 3] = height_scale
                        y_true[y][x][max_idx_anc * (5 + num_class) + 4] = 1.0
                        y_true[y][x][max_idx_anc * (5 + num_class) + int(cur_box[4])] = 1.0
                        if acn_assigned == len(anchor):
                            break
        return y_true

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

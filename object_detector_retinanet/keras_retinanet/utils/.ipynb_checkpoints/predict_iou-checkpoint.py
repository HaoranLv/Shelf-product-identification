"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

import csv
from PIL import Image
import datetime
import keras
from object_detector_retinanet.keras_retinanet.utils import EmMerger
from object_detector_retinanet.utils import create_folder, root_dir
from .visualization import draw_detections, draw_annotations

import numpy as np
import os
import csv
import cv2

def predict(
        generator,
        model,
        class_csv_root,
        score_threshold=0.05,
        max_detections=200,
        save_path=None,
        hard_score_rate=1.):
#     rd=root_dir()
    classes_list=[]
    with open(class_csv_root, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            classes_list.append(i[0])
    false_image=[]
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    csv_data_lst = []
    csv_data_lst.append(['image_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'hard_score', 'class'])
    result_dir = os.path.join(root_dir(), 'results')
    create_folder(result_dir)
    timestamp = datetime.datetime.utcnow()
    res_file = result_dir + '/detections_output_iou_{}_{}.csv'.format(hard_score_rate, timestamp)
    for i in range(generator.size()):
        try:
            image_name = os.path.join(generator.image_path(i).split(os.path.sep)[-2],
                                      generator.image_path(i).split(os.path.sep)[-1])
            print(generator.image_path(i))
            raw_image = generator.load_image(i)
            image = generator.preprocess_image(raw_image.copy())
            image, scale = generator.resize_image(image)

            # run network
            boxes, hard_scores, labels, soft_scores = model.predict_on_batch(np.expand_dims(image, axis=0))
            print(soft_scores.shape)
            soft_scores = np.max(soft_scores, axis=-1)
            #         soft_scores = np.mean(soft_scores, axis=-1)

            print(soft_scores.shape)
            #         soft_scores = np.squeeze(soft_scores, axis=-1)
            soft_scores = hard_score_rate * hard_scores + (1 - hard_score_rate) * soft_scores
            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(hard_scores[0, :] > score_threshold)[0]

            # select those scores
            scores = soft_scores[0][indices]
            hard_scores = hard_scores[0][indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            image_boxes = boxes[0, indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_hard_scores = hard_scores[scores_sort]
            image_labels = labels[0, indices[scores_sort]]
            print('image_labels', image_labels.shape)  # (xxx,)里面存的是排名前xxx的预测结果的label，int类型
            image_detections = np.concatenate(
                [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
            results = np.concatenate(
                [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_hard_scores, axis=1),
                 np.expand_dims(image_labels, axis=1)], axis=1)
            print('res', results.shape)  # (xxx,7)里面存的是排名前xxx的预测结果
            filtered_data, best_idx = EmMerger.merge_detections(image_name, results)  # 需要在这个地方返回到底那几个index是最终决定留下的obj
            print(best_idx)
            print(filtered_data.shape)  # (xx,11)这里的xx应该是代表一张图里最终筛选出的xx个obj
            filtered_boxes = []
            filtered_scores = []
            filtered_labels = []
            u = 0
            for ii, detection in filtered_data.iterrows():
                box = np.asarray([detection['x1'], detection['y1'], detection['x2'], detection['y2']])
                filtered_boxes.append(box)
                filtered_scores.append(detection['confidence'])
                filtered_labels.append('{}'.format(classes_list[image_labels[ii]]))
                #             filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
                row = [image_name, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                       detection['confidence'], detection['hard_score'], classes_list[image_labels[ii]]]
                csv_data_lst.append(row)

            if save_path is not None:
                create_folder(save_path)

                draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
                #             print(generator.label_to_name)
                draw_detections(raw_image, np.asarray(filtered_boxes), np.asarray(filtered_scores),
                                np.asarray(filtered_labels), color=(0, 0, 255))  # label_to_name=image_label[best_idx]

                cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
            #         print(all_detections)
            print('{}/{}'.format(i + 1, generator.size()), end='\r')
        except:
            image_name = os.path.join(generator.image_path(i).split(os.path.sep)[-2],
                                      generator.image_path(i).split(os.path.sep)[-1])
            print(image_name)
            false_image.append(image_name)

    # Save annotations csv file
    print('false image:',false_image)
    with open(res_file, 'w') as fl_csv:
        writer = csv.writer(fl_csv)
        writer.writerows(csv_data_lst)
    print("Saved output.csv file")

def predict_images(
        images_root_list,
        model,
        class_csv_root,
        score_threshold=0.05,
        max_detections=200,
        save_path=None,
        hard_score_rate=1.,
        res_root='/home/ec2-user/SageMaker/PDDS/SKU110K_CVPR19/customer'):
#     rd=root_dir()
    classes_list=[]
    with open(class_csv_root, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            classes_list.append(i[0])
    false_image=[]
    all_detections = [[None for i in range(len(classes_list))] for j in range(len(images_root_list))]
    csv_data_lst = []
    csv_data_lst.append(['image_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'hard_score', 'class'])
    result_dir = os.path.join(res_root, 'results')
    create_folder(result_dir)
    timestamp = datetime.datetime.utcnow()
    res_file = result_dir + '/detections_output_iou_{}_{}.csv'.format(hard_score_rate, timestamp)
    for i in range(len(images_root_list)):
        try:
            print(images_root_list[i])
            image_name = os.path.join(images_root_list[i].split(os.path.sep)[-2],
                                      images_root_list[i].split(os.path.sep)[-1])
            raw_image = read_image_bgr(images_root_list[i])
            image = preprocess_image(raw_image.copy())
            image, scale = resize_image(image)
            # run network 
            boxes, hard_scores, labels, soft_scores = model.predict_on_batch(np.expand_dims(image, axis=0))
#             print(soft_scores.shape)
            soft_scores = np.max(soft_scores, axis=-1)
            #         soft_scores = np.mean(soft_scores, axis=-1)

#             print(soft_scores.shape)
            #         soft_scores = np.squeeze(soft_scores, axis=-1)
            soft_scores = hard_score_rate * hard_scores + (1 - hard_score_rate) * soft_scores
            # correct boxes for image scale
            boxes /= scale
            # select indices which have a score above the threshold
            indices = np.where(hard_scores[0, :] > score_threshold)[0]

            # select those scores
            scores = soft_scores[0][indices]
            hard_scores = hard_scores[0][indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            image_boxes = boxes[0, indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_hard_scores = hard_scores[scores_sort]
            image_labels = labels[0, indices[scores_sort]]
#             print('image_labels', image_labels.shape)  # (xxx,)里面存的是排名前xxx的预测结果的label，int类型
            image_detections = np.concatenate(
                [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
            results = np.concatenate(
                [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_hard_scores, axis=1),
                 np.expand_dims(image_labels, axis=1)], axis=1)
#             print('res', results.shape)  # (xxx,7)里面存的是排名前xxx的预测结果
            filtered_data, best_idx = EmMerger.merge_detections(image_name, results)  # 需要在这个地方返回到底那几个index是最终决定留下的obj
#             print(best_idx)
#             print(filtered_data.shape)  # (xx,11)这里的xx应该是代表一张图里最终筛选出的xx个obj
            filtered_boxes = []
            filtered_scores = []
            filtered_labels = []
            u = 0
            for ii, detection in filtered_data.iterrows():
                box = np.asarray([detection['x1'], detection['y1'], detection['x2'], detection['y2']])
                filtered_boxes.append(box)
                filtered_scores.append(detection['confidence'])
                filtered_labels.append('{}'.format(classes_list[image_labels[ii]]))
                #             filtered_labels.append('{0:.2f}'.format(detection['hard_score']))
                row = [image_name, detection['x1'], detection['y1'], detection['x2'], detection['y2'],
                       detection['confidence'], detection['hard_score'], classes_list[image_labels[ii]]]
                csv_data_lst.append(row)
            if save_path is not None:
                create_folder(save_path)

#                 draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
                #             print(generator.label_to_name)
                draw_detections(raw_image, np.asarray(filtered_boxes), np.asarray(filtered_scores),
                                np.asarray(filtered_labels), color=(0, 0, 255))  # label_to_name=image_label[best_idx]

                cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)

            # copy detections to all_detections
#             for label in range(generator.num_classes()):
#                 all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
#             #         print(all_detections)
            print('{}/{}'.format(i + 1, len(images_root_list)), end='\r')
        except:
            print(images_root_list[i])
            false_image.append(images_root_list[i])

    # Save annotations csv file
    print('false image:',false_image)
    with open(res_file, 'w') as fl_csv:
        writer = csv.writer(fl_csv)
        writer.writerows(csv_data_lst)
    print("Saved output.csv file")

def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def preprocess_image(x):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        data_format:           Same as for keras.preprocessing.image.apply_transform
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        data_format          = None,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

        if data_format is None:
            data_format = keras.backend.image_data_format()
        self.data_format = data_format

        if data_format == 'channels_first':
            self.channel_axis = 0
        elif data_format == 'channels_last':
            self.channel_axis = 2
        else:
            raise ValueError("invalid data_format, expected 'channels_first' or 'channels_last', got '{}'".format(data_format))

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    if params.channel_axis != 2:
        image = np.moveaxis(image, params.channel_axis, 2)

    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )

    if params.channel_axis != 2:
        output = np.moveaxis(output, 2, params.channel_axis)
    return output


def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale
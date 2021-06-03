# -*- coding: utf-8 -*-

import os
import json
import boto3
import argparse
import os
import sys
from PIL import Image
import keras
import tensorflow as tf
from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.preprocessing.csv_generator import CSVGenerator
from object_detector_retinanet.keras_retinanet.utils.predict_iou import predict_aws
from object_detector_retinanet.keras_retinanet.utils.keras_version import check_keras_version
from object_detector_retinanet.utils import image_path, annotation_path, root_dir
from keras import backend as K
import warnings
import numpy as np
import cv2
import time
warnings.filterwarnings("ignore",category=FutureWarning)
import matplotlib.pyplot as plt


import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)


import flask

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

# # Set up model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Darknet("yolov3-custom.cfg", img_size=416).to(device)
# # Load checkpoint weights
# weights_path = 'yolov3_ckpt_1580.pth'
# model.load_state_dict(torch.load(weights_path,map_location=device))

# model.eval()  # Set in evaluation mode

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# load model
hard_score_rate = 0.5
print ("hard_score_rate={}".format(hard_score_rate))
use_cpu = False
gpu_num = str(0)
backbone='resnet50'
convert_model=1
if use_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(666)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
global sess
global graph
sess = tf.Session()
K.set_session(sess)
# keras.backend.tensorflow_backend.set_session(get_session())
print('Loading model, this may take a second...')
model = models.load_model('./model/iou_resnet50_csv_01.h5', backbone_name=backbone, convert=convert_model, nms=False)
graph = tf.get_default_graph()
image_path='./test.jpg'
class_csv_root = './object_detector_retinanet/keras_retinanet/bin/class_mappings.csv'
print('Finish loading!')
print('12')
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

def infer(model, data_np, class_csv_root):
#     os.makedirs("output", exist_ok=True)
    print("\nPerforming object detection:")
    raw_image = data_np
    image = preprocess_image(raw_image.copy())
    image, scale = resize_image(image)
    with graph.as_default():# 使用保存后的graph做inference
        K.set_session(sess)
        detections = predict_aws(
            image, 
            scale,
            raw_image,
            model,
            class_csv_root,
            score_threshold=0.1,
            max_detections=200,
            hard_score_rate=hard_score_rate)
    return detections

def infer1(model, image_path, class_csv_root):
#     os.makedirs("output", exist_ok=True)
    print("\nPerforming object detection:")
    raw_image = read_image_bgr(image_path)
    image = preprocess_image(raw_image.copy())
    image, scale = resize_image(image)
    with graph.as_default():# 使用保存后的graph做inference
        K.set_session(sess)
        detections = predict_aws(
            image, 
            scale,
            raw_image,
            model,
            class_csv_root,
            score_threshold=0.1,
            max_detections=200,
            hard_score_rate=hard_score_rate)
    return detections


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    tic = time.time()
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    print("================ INVOCATIONS =================")
    print ("<<<< title: ", flask.request)
    
    data = flask.request.data
    print("len(data)={}".format(len(data)))
    data_np = np.fromstring(data, dtype=np.uint8)
    print("data_np.shape={}".format(str(data_np.shape)))
    print(' '.join(['{:x}'.format(d) for d in data_np[:20].tolist()]), flush=True)
    data_np = cv2.imdecode(data_np, cv2.IMREAD_UNCHANGED)
    print ('data_np: ', data_np)
    data_np = cv2.cvtColor(data_np, cv2.COLOR_RGB2BGR)
#     image_path='./test.jpg'
    #print ("<<<< flask.request.data.content_type", flask.request.data.content_type)
    print ("<<<< flask.request.content_type", flask.request.content_type)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print ("<<<< gpu 推理 device")
    label = infer(model,data_np , class_csv_root)
#     label = infer1(model, image_path , class_csv_root)
    toc = time.time()
    print(f"0 - invocations: {(toc - tic) * 1000.0} ms")

    inference_result = {
            'result': label
        }
    _payload = json.dumps(inference_result, ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')

    #if flask.request.content_type == 'image/jpeg':
       

    #else:
     #   return flask.Response(response='This predictor only supports JSON data and JPEG image data',
                 #             status=415, mimetype='text/plain')
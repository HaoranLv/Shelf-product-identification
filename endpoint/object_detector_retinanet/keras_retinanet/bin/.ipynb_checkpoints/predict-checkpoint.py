#!/usr/bin/env python

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

import argparse
import os
import sys

import keras
import tensorflow as tf


from object_detector_retinanet.keras_retinanet import models
from object_detector_retinanet.keras_retinanet.preprocessing.csv_generator import CSVGenerator
from object_detector_retinanet.keras_retinanet.utils.predict_iou import predict, predict_images
from object_detector_retinanet.keras_retinanet.utils.keras_version import check_keras_version
from object_detector_retinanet.utils import image_path, annotation_path, root_dir
abs_root='/home/ec2-user/SageMaker/PDDS/SKU110K_CVPR19/customer'

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    if args.dataset_type == 'csv':
        print(args.annotations)
        print(args.base_dir)
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            base_dir=args.base_dir
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.'/home/ec2-user/SageMaker/PDDS/SKU110K_CVPR19/object_detector_retinanet/keras_retinanet/bin/class_mappings.csv'
    """
    parser = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')
    
    img_parser = subparsers.add_parser('single_image')
    img_parser.add_argument('--image_path', help='Path to single image (ie. /tmp/COCO).')
    img_parser.add_argument('--class_csv_root', help='Path to CSV file containing classes for evaluation.',
                            default='/home/ec2-user/SageMaker/PDDS/SKU110K_CVPR19/object_detector_retinanet/keras_retinanet/bin/class_mappings.csv')
    
    imgs_parser = subparsers.add_parser('images')
    imgs_parser.add_argument('--image_path', help='Path to images directory (ie. /tmp/COCO).')
    imgs_parser.add_argument('--class_csv_root', help='Path to CSV file containing classes for evaluation.',
                            default='/home/ec2-user/SageMaker/PDDS/SKU110K_CVPR19/object_detector_retinanet/keras_retinanet/bin/class_mappings.csv')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    data_dir = annotation_path(abs_root=abs_root, datafolder = 'customer')
    args_annotations = data_dir + '/annotations_test.csv'

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('--annotations', help='Path to CSV file containing annotations for evaluation.',
                            default=args_annotations)
    csv_parser.add_argument('--classes', help='Path to a CSV file containing class label mapping.',
                            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_mappings.csv'))
    parser.add_argument('--hard_score_rate', help='', default="0.5")
    csv_parser.add_argument('--class_csv_root', help='Path to CSV file containing classes for evaluation.',
                            default='/home/ec2-user/SageMaker/PDDS/SKU110K_CVPR19/object_detector_retinanet/keras_retinanet/bin/class_mappings.csv')

    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument('--base_dir', help='Path to base dir for CSV file.',
                        default=image_path(abs_root=abs_root, datafolder = 'customer'))
    parser.add_argument('--convert-model',
                        help='Convert the model to an inference model (ie. the input is a training model).', type=int,
                        default=1)

    parser.add_argument('--backbone', help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).',
                        default=0.1, type=float)
    parser.add_argument('--iou-threshold', help='IoU Threshold to count for a positive detection (defaults to 0.5).',
                        default=0.75, type=float)
    parser.add_argument('--save-path', help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int,
                        default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.',
                        type=int, default=1333)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    if args.hard_score_rate:
        hard_score_rate = float(args.hard_score_rate.lower())
    else:
        hard_score_rate = 0.5
    print ("hard_score_rate={}".format(hard_score_rate))
    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    use_cpu = False

    if args.gpu:
        gpu_num = args.gpu
    else:
        gpu_num = str(0)

    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(666)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # load the model
    print('Loading model, this may take a second...')
    model = models.load_model(os.path.join(root_dir(abs_root=abs_root, datafolder = 'customer'), args.model), backbone_name=args.backbone, convert=args.convert_model, nms=False)
    # create the generator
    if args.dataset_type=='csv':
        generator = create_generator(args)
        predict(
        generator,
        model,
        args.class_csv_root,
        score_threshold=args.score_threshold,
        save_path=os.path.join(root_dir(abs_root=abs_root, datafolder = 'customer'), 'res_images_iou'),
        hard_score_rate=hard_score_rate
    )
        
    elif args.dataset_type=='single_image':
        image_root=[args.image_path]
        predict_images(
        image_root,
        model,
        args.class_csv_root,
        score_threshold=args.score_threshold,
        save_path=os.path.join(root_dir(abs_root=abs_root, datafolder = 'customer'), 'res_images_iou'),
        hard_score_rate=hard_score_rate,
        res_root=abs_root
    )
        
    elif args.dataset_type=='images':
        image_root=[]
        rootdir = args.image_path
        l =os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(len(l)):
            path = os.path.join(rootdir, l[i])
            image_root.append(path)
        predict_images(
        image_root,
        model,
        args.class_csv_root,
        score_threshold=args.score_threshold,
        save_path=os.path.join(root_dir(abs_root=abs_root, datafolder = 'customer'), 'res_images_iou'),
        hard_score_rate=hard_score_rate,
        res_root=abs_root)


if __name__ == '__main__':
    main()

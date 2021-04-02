# Python
import logging
import pathlib
import os
import sys

# lib
import numpy as np
from PIL import Image
import tensorflow as tf

from rpi_deep_pantilt import __path__ as rpi_deep_pantilt_path
from rpi_deep_pantilt.detect.util.label import create_category_index_from_labelmap
from rpi_deep_pantilt.detect.util.visualization import visualize_boxes_and_labels_on_image_array

LABELS = ['robot', 'ball', 'goal']


class SSDMobileDet_SSL_EdgeTPU_Quant(object):

    EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
    PATH_TO_LABELS = rpi_deep_pantilt_path[0] + '/data/ssldataset_label_map.pbtxt'

    def __init__(
        self,
        base_url='https://github.com/leigh-johnson/rpi-deep-pantilt/releases/download/v1.0.0/',
        model_name='ssdlite_mobilenet_edgetpu_coco_quant',
        input_shape=(224, 224),
        min_score_thresh=0.50,
        tflite_model_file='mobilenet_70_30_raw_whole_3000_edgetpu.tflite',
        model_path_dir="/home/ssl/Documents/msc-project/mobilenet/exported-models/tf1/mobilenet_70_30_whole_50000/mobilenet_70_30_whole_50000_edgetpu.tflite"
    ):

        self.base_url = base_url
        self.model_name = model_name
        self.model_file = model_name + '.tar.gz'
        self.model_url = base_url + self.model_file
        self.tflite_model_file = tflite_model_file

        # self.model_dir = tf.keras.utils.get_file(
        #     fname=self.model_file,
        #     origin=self.model_url,
        #     untar=True,
        #     cache_subdir='models'
        # )

        self.min_score_thresh = min_score_thresh

        self.model_path = model_path_dir
        
        try:
            from tflite_runtime import interpreter as coral_tflite_interpreter
        except ImportError as e:
            logging.error(e)
            logging.error('Please install Edge TPU tflite_runtime:')
            logging.error(
                '$ pip install 	https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl')
            sys.exit(1)

        self.tflite_interpreter = coral_tflite_interpreter.Interpreter(
            model_path=self.model_path,
            experimental_delegates=[
                tf.lite.experimental.load_delegate(self.EDGETPU_SHARED_LIB)
            ]
        )

        self.tflite_interpreter.allocate_tensors()

        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()

        self.category_index = create_category_index_from_labelmap(
            self.PATH_TO_LABELS, use_display_name=True)

        logging.info(
            f'loaded labels from {self.PATH_TO_LABELS} \n {self.category_index}')

        logging.info(f'initialized model {model_name} \n')
        logging.info(
            f'model inputs: {self.input_details} \n {self.input_details}')
        logging.info(
            f'model outputs: {self.output_details} \n {self.output_details}')

    def label_to_category_index(self, labels):
        # @todo :trashfire:
        return tuple(map(
            lambda x: x['id'],
            filter(
                lambda x: x['name'] in labels, self.category_index.values()
            )
        ))

    def set_model_path(self, m_path):
        self.model_path = m_path

    def label_display_name_by_idx(self, idx):
        return self.category_index[idx]['display_name']

    def filter_tracked(self, prediction, label_idxs):
        '''
            Zero predictions not in list of label_idxs
        '''
        return {
            'detection_boxes': prediction.get('detection_boxes'),
            'detection_classes': prediction.get('detection_classes'),
            'detection_scores':
                np.array(
                    [v if prediction.get('detection_classes')[i] in label_idxs
                     else 0.0
                     for i, v in enumerate(prediction.get('detection_scores'))])
        }

    def create_overlay(self, image_np, output_dict, video_path):

        if not video_path:
            image_np = image_np.copy()

        # draw bounding boxes
        visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=self.min_score_thresh,
            max_boxes_to_draw=None
        )

        img = Image.fromarray(image_np)

        return img.tobytes()

    def predict(self, image, is_yolo_det, input_size):
        '''
            image - np.array (3 RGB channels)

            returns <dict>
                {
                    'detection_classes': int64,
                    'num_detections': int64
                    'detection_masks': ...
                }
        '''

        if is_yolo_det:
            from rpi_deep_pantilt.detect.util.yolov4 import decode, filter_boxes
            import cv2
            image_data = cv2.resize(image, (input_size, input_size))
            image_data = image_data / 255.
            input_tensor = []
            for i in range(1):
                input_tensor.append(image_data)
            input_tensor = np.asarray(input_tensor).astype(np.float32)
        else:
            image = np.asarray(image)
            # normalize 0 - 255 RGB to values between (-1, 1)
            #image = (image / 128.0) - 1

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.

            input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)

            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        self.tflite_interpreter.set_tensor(
            self.input_details[0]['index'], input_tensor)

        self.tflite_interpreter.invoke()

        # TFLite_Detection_PostProcess custom op node has four outputs:
        # detection_boxes: a float32 tensor of shape [1, num_boxes, 4] with box
        # locations
        # detection_classes: a float32 tensor of shape [1, num_boxes]
        # with class indices
        # detection_scores: a float32 tensor of shape [1, num_boxes]
        # with class scores
        # num_boxes: a float32 tensor of size 1 containing the number of detected
        # boxes

        # Without the PostProcessing ops, the graph has two outputs:
        #    'raw_outputs/box_encodings': a float32 tensor of shape [1, num_anchors, 4]
        #     containing the encoded box predictions.
        #    'raw_outputs/class_predictions': a float32 tensor of shape
        #     [1, num_anchors, num_classes] containing the class scores for each anchor
        #     after applying score conversion.

        pred = [self.tflite_interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        if is_yolo_det:
            from rpi_deep_pantilt.detect.util.yolov4 import get_anchors
            STRIDES = np.array([16, 32])
            ANCHORS = get_anchors([23,27, 37,58, 81,82, 81,82, 135,169, 344,319])
            XYSCALE = [1.05, 1.05]
            bbox_tensors = []
            prob_tensors = []
            for i, fm in enumerate(pred):
                if i == 0:
                    output_tensors = decode(pred[1], input_size // 16, 3, STRIDES, ANCHORS, i, XYSCALE)
                else:
                    output_tensors = decode(pred[0], input_size // 32, 3, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(output_tensors[0])
                prob_tensors.append(output_tensors[1])
            pred_bbox = tf.concat(bbox_tensors, axis=1)
            pred_prob = tf.concat(prob_tensors, axis=1)
            pred = (pred_bbox, pred_prob)
            box_data, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape = tf.constant([input_size,input_size]))
            box_data, score_data, class_data, num_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(box_data, (tf.shape(box_data)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.25
            )
            pred_bbox = [box_data.numpy(), score_data.numpy(), class_data.numpy(), num_detections.numpy()]

        else:
            box_data = tf.convert_to_tensor(pred[0])
            class_data = tf.convert_to_tensor(pred[1])
            score_data = tf.convert_to_tensor(pred[2])
            num_detections = tf.convert_to_tensor(pred[3])

        class_data = tf.squeeze(
            class_data, axis=[0]).numpy().astype(np.int64) + 1
        box_data = tf.squeeze(box_data, axis=[0]).numpy()
        score_data = tf.squeeze(score_data, axis=[0]).numpy()

        # logging.info(box_data)
        # logging.info(class_data)
        # logging.info(score_data)

        return {
            'detection_boxes': box_data,
            'detection_classes':  class_data,
            'detection_scores': score_data,
            'num_detections': len(num_detections)
        }

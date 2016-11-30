# coding: utf-8

# # Detection with SSD
#
# In this example, we will load a SSD model and use it to detect objects.

# ### 1. Setup
#
# * First, Load necessary libs and set up caffe and caffe_root

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import os
import sys

caffe_root = './'  # this file is expected to be in {caffe_root}
os.chdir(caffe_root)
sys.path.insert(0, 'python')

import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2


class SSDDetector:
    def __init__(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        labelmap_file = open('data/coco/labelmap_coco_minsun.prototxt', 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(labelmap_file.read()), self.labelmap)

        # * Load the net in the test phase for inference, and configure input preprocessing.
        model_def = 'models/VGGNet/coco/SSD_300x300/deploy.prototxt'
        model_weights = 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_240000.caffemodel'

        self.net = caffe.Net(model_def,  # defines the structure of the model
                        model_weights,  # contains the trained weights
                        caffe.TEST)  # use test mode (e.g., don't perform dropout)

        # input pre-processing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

        # ### 2. SSD detection
        # Load an image. Set net to batch size of 1
        image_resize = 300
        self.net.blobs['data'].reshape(1, 3, image_resize, image_resize)

    def get_labelname(self, labels):
        num_labels = len(self.labelmap.item)
        labelnames = []
        if type(labels) is not list:
            labels = [labels]
        for label in labels:
            found = False
            for i in xrange(0, num_labels):
                if label == self.labelmap.item[i].label:
                    found = True
                    labelnames.append(self.labelmap.item[i].display_name)
                    break
            if not found:
                labelnames.append(None)
        return labelnames

    @staticmethod
    def rgb_to_caffe_input(frame):
        img = frame / 255.0
        return img[:, :, (2, 1, 0)]

    def detect(self, frame=None, conf_threshold=0.3):
        # convert format
        image = self.rgb_to_caffe_input(frame)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        begin_time = time.time()
        detections = self.net.forward()['detection_out']
        end_time = time.time()
        print('time %.2f' % (end_time - begin_time))

        # Parse the outputs.
        det_label = detections[0, 0, :, 1]
        det_conf = detections[0, 0, :, 2]
        det_xmin = detections[0, 0, :, 3]
        det_ymin = detections[0, 0, :, 4]
        det_xmax = detections[0, 0, :, 5]
        det_ymax = detections[0, 0, :, 6]

        # Get detections with confidence higher than 0.3.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_threshold]

        top_conf = det_conf[top_indices]
        top_label_indices = [int(x) for x in det_label[top_indices].tolist()]
        top_labels = self.get_labelname(top_label_indices)
        top_xmin = det_xmin[top_indices] * image.shape[1]
        top_ymin = det_ymin[top_indices] * image.shape[0]
        top_xmax = det_xmax[top_indices] * image.shape[1]
        top_ymax = det_ymax[top_indices] * image.shape[0]

        ret_array = list()
        for i in range(len(top_conf)):
            det = {'label': top_labels[i], 'label_id': top_label_indices[i], 'conf': top_conf[i], 'id': i+1,
               'x1': top_xmin[i], 'y1': top_ymin[i],
               'x2': top_xmax[i], 'y2': top_ymax[i]}
            ret_array.append(det)
        return ret_array

    @staticmethod
    def plot(image, det):
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'

        top_conf = det.get('conf')
        top_labels = det.get('label')
        top_label_id = det.get('label_id')
        xmin = det.get('xmin')
        ymin = det.get('ymin')
        xmax = det.get('xmax')
        ymax = det.get('ymax')

        # * Plot the boxes
        plot_enable = True
        if plot_enable:
            colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            currentAxis = plt.gca()

        obj_i = 0
        frame_i = 0
        for i in xrange(top_conf.shape[0]):
            width = xmax - xmin + 1
            height = ymax - ymin + 1
            score = top_conf[i]
            label_name = top_labels[i]
            label = top_label_id[i]
            display_txt = '%s: %.2f' % (label_name, score)
            # print(display_txt, end=' ')

            # Min Sun dataset format

            # skip class not in Min Sun dataset
            if label_name is not None:
                obj_i += 1
                det_format = '%06d\t%d\t"%s"\t%d\t%d\t%d\t%d\t%f' % (
                    frame_i, obj_i, label_name, xmin, ymin, xmax, ymax, score)
                print(det_format)
            else:
                det_format = '%06d\t%d\t"%s"\t%d\t%d\t%d\t%d\t%f' % (
                    frame_i, 0, 'Other', xmin, ymin, xmax, ymax, score)
                print(det_format)

            if plot_enable:
                coords = (xmin, ymin), width, height
                color = colors[label]
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

            print()
        plt.show()


def main():
    # I/O Path
    if len(sys.argv) < 2:
        print('run_ssd.py <input_video_path>')
        raise Exception

    input_image_path = sys.argv[1]  # file or path both ok
    frame = cv2.imread(input_image_path)  # BGR

    ssd_detector = SSDDetector()
    det = ssd_detector.detect(frame, 0.6)
    print(det)
    ssd_detector.plot(frame, det)


if __name__ == '__main__':
    main()

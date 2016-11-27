
# coding: utf-8

# # Detection with SSD
# 
# In this example, we will load a SSD model and use it to detect objects.

# ### 1. Setup
# 
# * First, Load necessary libs and set up caffe and caffe_root

# In[1]:
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import cv2
import os
import sys
import shutil

caffe_root = './'  # this file is expected to be in {caffe_root}
os.chdir(caffe_root)
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# I/O Path
if len(sys.argv) < 2:
	print('run_ssd.py <input_video_path>')
	raise Exception

input_video = sys.argv[1]  # file or path both ok
output_dir = 'ssd_output'

# Generate detection result image (for visualization)  It is much slower than pure SSD computation
plot_enable = False
if len(sys.argv) >= 3 and (sys.argv[2] == 'plot'):
	plot_enable = True
	
'''
if os.path.exists(output_dir):
	shutil.rmtree(output_dir)
os.makedirs(output_dir)
'''

# * Load LabelMap.

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC / MSCOCO labels

# labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
labelmap_file = 'data/coco/labelmap_coco_minsun.prototxt'

file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        if not found:
            labelnames.append(None)
    return labelnames

# * Load the net in the test phase for inference, and configure input preprocessing.

# In[3]:

# model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
# model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'

model_def = 'models/VGGNet/coco/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_240000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# ### 2. SSD detection

# * Load an image.

# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'



def ssd_detect_single_video(video_file_path):

	video_name = os.path.splitext(os.path.basename(video_file_path))[0]

	# Open video file as cap
	cap = cv2.VideoCapture(video_file_path)
	if not cap.isOpened():
		raise IOError

	frame_i = 0

	det_dir = os.path.join(output_dir, 'det')
	if not os.path.exists(det_dir):
		os.makedirs(det_dir)
	

	video_dir = os.path.join(output_dir, 'videos', video_name)
	if not os.path.exists(video_dir) and plot_enable:
		os.makedirs(video_dir)
	
	with open(os.path.join(det_dir,	video_name + '_det.txt'), 'w') as det_file:
		while True:
			frame_i += 1
			# image = caffe.io.load_image(path)
			
			ret, frame = cap.read()
			if frame is None:
				break;

			# convert format
			frame = frame / 255.
			frame = frame[:,:,(2,1,0)]
			image = frame
			
			# * Run the net and examine the top_k results

			transformed_image = transformer.preprocess('data', image)
			net.blobs['data'].data[...] = transformed_image

			# Forward pass.
			begin_time = time.time()
			detections = net.forward()['detection_out']
			end_time = time.time()
			# print('frame %d takes %.2f ms' % (frame_i, (end_time - begin_time) * 1000), end=': ')

			# Parse the outputs.
			det_label = detections[0,0,:,1]
			det_conf = detections[0,0,:,2]
			det_xmin = detections[0,0,:,3]
			det_ymin = detections[0,0,:,4]
			det_xmax = detections[0,0,:,5]
			det_ymax = detections[0,0,:,6]
			
			# Get detections with confidence higher than 0.3.
			top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]
			
			top_conf = det_conf[top_indices]
			top_label_indices = det_label[top_indices].tolist()
			top_labels = get_labelname(labelmap, top_label_indices)
			top_xmin = det_xmin[top_indices]
			top_ymin = det_ymin[top_indices]
			top_xmax = det_xmax[top_indices]
			top_ymax = det_ymax[top_indices]
			

			# * Plot the boxes
			if plot_enable:
				colors = plt.cm.hsv(np.linspace(0, 1, 81)).tolist()
				plt.imshow(image)
				currentAxis = plt.gca()

			obj_i = 0
			for i in xrange(top_conf.shape[0]):
				xmin = int(round(top_xmin[i] * image.shape[1]))
				ymin = int(round(top_ymin[i] * image.shape[0]))
				xmax = int(round(top_xmax[i] * image.shape[1]))
				ymax = int(round(top_ymax[i] * image.shape[0]))
				width = xmax-xmin+1
				height = ymax-ymin+1
				score = top_conf[i]
				label = int(top_label_indices[i])
				label_name = top_labels[i]
				display_txt = '%s: %.2f'%(label_name, score)
				# print(display_txt, end=' ')
				
				# Min Sun dataset format


				# skip class not in Min Sun dataset
				if label_name is not None:
					obj_i += 1
					det_format = '%06d\t%d\t"%s"\t%d\t%d\t%d\t%d\t%f' % (frame_i, obj_i, label_name, xmin, ymin, xmax, ymax, score)
					print(det_format)
					det_file.write(det_format + '\n')
				else:
					det_format = '%06d\t%d\t"%s"\t%d\t%d\t%d\t%d\t%f' % (frame_i, 0, 'Other', xmin, ymin, xmax, ymax, score)
					print(det_format)
				
				if plot_enable:
					coords = (xmin, ymin), width, height 
					color = colors[label]
					currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
					currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
			
			if plot_enable:		
				plt.savefig(os.path.join(video_dir, ('det_%06d.jpg' % frame_i)), bbox_inches='tight')
				plt.clf() # clear content
				
			# print()

			
def ssd_detect_video(input_video_path):

	if os.path.isdir(input_video_path):
		for	file in sorted([f for f in os.listdir(input_video_path)]):
			file_path = os.path.join(input_video_path, file)
			print('process', file_path)
			ssd_detect_single_video(file_path)
			
	elif os.path.isfile(input_video_path):
		print('process', input_video_path)
		ssd_detect_single_video(input_video_path)
		
	else:
		raise IOError


ssd_detect_video(input_video)

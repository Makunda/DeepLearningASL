#!/usr/bin/env python2.7
"""
MultiThreaded Mask R-CNN
Mask R-CNN model launch file.

Copyright (c) 2019.
Licensed under the MIT License (see LICENSE for details)
Written by Hugo JOBY
"""
import random, time
import threading


import os
import sys
import random
import math
import numpy as np
import scipy.misc

import cv2
import imageio
import visualize

import coco
import utils
import model as modellib

import time
import settings
import signal

class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_MIN_CONFIDENCE = 0.5



class DeeplearningManager(threading.Thread):
	"""docstring for deeplearning"""
	def __init__(self, frame, bboxes, lock_bboxes):
		threading.Thread.__init__(self)
		self.shutdown_flag = threading.Event()

		self.bboxes = bboxes
		self.frame = frame
		self.newbboxes = []
		self.lock_bboxes = lock_bboxes

		# Root directory of the project
		self.ROOT_DIR = os.getcwd()

		# Directory of images to run detection on
		self.IMAGE_DIR = os.path.join(self.ROOT_DIR, "images")
		self.VIDEO_DIR = os.path.join(self.ROOT_DIR, "video")

        # Directory to save logs and trained model
		self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")

		self.config = InferenceConfig()
		self.config.print_conf()



		self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)
		# Load weights trained on MS-COCO
		# Path to trained weights file
		# Download this file and place in the root of your 
		# project (See README file for details)
		self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
		self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)
		# COCO Class names
		# Index of the class in the list is its ID. For example, to get ID of
		# the teddy bear class, use: class_names.index('teddy bear')
		self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
					'bus', 'train', 'truck', 'boat', 'traffic light',
					'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
					'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
					'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
					'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
					'kite', 'baseball bat', 'baseball glove', 'skateboard',
					'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
					'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
					'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
					'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
					'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
					'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
					'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
					'teddy bear', 'hair drier', 'toothbrush']

		
	def convert_model_result(self, results):
		return_list = []
		for roi, mask, class_id, score in zip(results['rois'], results['masks'], results['class_ids'], results['scores']):
			return_list.append({'roi' : roi, 'mask' : mask, 'class_id' : class_id, 'score': score})
		return return_list


	def return_image_rcnn(self, image):

		if image is None:
			print("Error Image has no type..")
			return 0
		else:
			print("Image Type :"+str(type(image)))
		    # Visualize results
			results = self.model.detect([image], verbose=0)
			# Visualize result
			
			settings.GLOBAL_BBOXES = self.convert_model_result(results[0])

			return 1

	

	def run(self):
		while not settings.GLOBAL_SHUTDOWN_FLAG.is_set():
			settings.GLOBAL_FRAME_LOCK.acquire()
			print("Lock acquired dp")
			image = settings.GLOBAL_FRAME.copy()
			settings.GLOBAL_FRAME_LOCK.release()
			print("Lock released dp")

			if image is None:
				print("Error Image has no type..")
				print("Error in deeplearning..")
				settings.GLOBAL_BBOXES = []
				time.sleep(0.1)
				#settings.GLOBAL_SHUTDOWN_FLAG.set()
			else:
				print("Lancement deeplearning")
				results = self.model.detect([image], verbose=0)
				print("Fin deeplearning")

				#cv2.imwrite('messigray.png',image)
				# print("Lock needed : DL")
				# self.lock_bboxes.acquire()
				settings.GLOBAL_BBOXES = self.convert_model_result(results[0])
				print("Number of result : "+str(len(settings.GLOBAL_BBOXES)))
				#print("After conversion : ")
				#print(type(settings.GLOBAL_BBOXES))
				#settings.GLOBAL_BBOXES, settings.GLOBAL_BBOXES_MASK, settings.GLOBAL_BBOXES_ID, settings.GLOBAL_BBOXES_SCORE = r['rois'], r['masks'], r['class_ids'], r['scores']
				# self.lock_bboxes.release()
				# print("Lock released : DL")
		return 1

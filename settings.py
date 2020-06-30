#!/usr/bin/env python2.7
"""
MultiThreaded Mask R-CNN
Mask R-CNN model launch file.

Copyright (c) 2019.
Licensed under the MIT License (see LICENSE for details)
Written by Hugo JOBY
"""

import threading

"""
FORMAT IN FILES
#CV2 ROI FORMAT = x,y,w,h
#KERAS BBOX FORMAT = y1, x1, y2, x2



"""

def init():
	#GLOBAL VideoReader
	global GLOBAL_FRAME
	global GLOBAL_OLD_FRAME
	global GLOBAL_FRAME_TRESHOLDED

	#GLOBAL DEEPLEARNING
	global GLOBAL_BBOXES 

	#GLOBAL TRACKING
	global GLOBAL_TRACKED_BBOXES

	#GLOBAL PREDICTION
	global GLOBAL_PREDICTED_BBOXES
	global GLOBAL_PREDICTED_TRAJECTORY

	global GLOBAL_CLASS_NAME
	
	#Locks 
	global GLOBAL_SHUTDOWN_FLAG
	global GLOBAL_THREAD_LOCK
	global GLOBAL_FRAME_LOCK
	global GLOBAL_TRESHOLDED_FRAME_LOCK

	#Event in threads
	global GLOBAL_SHUTDOWN_FLAG

	#Global parameters
	global GLOBAL_FRAMERATE

	GLOBAL_FRAMERATE = 0.1

	GLOBAL_SHUTDOWN_FLAG = threading.Event()
	GLOBAL_THREAD_LOCK = threading.Lock()
	GLOBAL_FRAME_LOCK = threading.Lock()
	GLOBAL_TRESHOLDED_FRAME_LOCK = threading.Lock()

	GLOBAL_FRAME = None
	GLOBAL_OLD_FRAME = None
	GLOBAL_FRAME_TRESHOLDED = None

	#({'roi' : roi, 'mask' : mask, 'class_id' : class_id, 'score': score})
	#GLOBAL_BBOXES[i]['roi']
	GLOBAL_BBOXES = []
	GLOBAL_TRACKED_BBOXES = []	
	GLOBAL_PREDICTED_BBOXES = []


	GLOBAL_PREDICTED_TRAJECTORY = []

	GLOBAL_CLASS_NAME =['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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

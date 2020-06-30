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


import cv2
import imageio
import visualize

import os
import sys
import random
import math
import numpy as np
import scipy.misc
import settings
import signal




class TrackerManager(threading.Thread):
	"""docstring for TrackerManager"""
	def __init__(self, frame, bboxes, lock_bboxes):
		threading.Thread.__init__(self)
		self.shutdown_flag = threading.Event()

		self.lock_bboxes = lock_bboxes
		self.bboxes = settings.GLOBAL_BBOXES
		self.frame = settings.GLOBAL_FRAME
		self.newbboxes = []

		self.multiTracker = []

		self.trackerType = 'CSRT'

		for i, bbox in enumerate(self.bboxes):
			bbox = int(bbox[1]), int(bbox[0]), abs(int(bbox[3]) - int(bbox[1])), abs(int(bbox[2]) - int(bbox[0]))
			bbox = tuple(bbox)
			bboxes[i] = bbox
			tracker = self.createTrackerByName(self.trackerType) 
			self.multiTracker.add(tracker, settings.GLOBAL_FRAME, bbox)

	def setTrackerype(self, trackerType):
		self.trackerType = trackerType

	def getTrackerype(self, trackerType):
		return self.trackerType


	def selective_search(self, im):
		newHeight = 200
		newWidth = int(im.shape[1]*200/im.shape[0])
		im = cv2.resize(im, (newWidth, newHeight))    
	
		# create Selective Search Segmentation Object using default parameters
		ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	 
		# set input image on which we will run segmentation
		ss.setBaseImage(im)
		ss.switchToSelectiveSearchFast()
	 
		# run selective search segmentation on input image
		rects = ss.process()
		return rects
	
	#KCF not working
	def createTrackerByName(self, trackerType):
		trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
		  # Create a tracker based on tracker name
		if self.trackerType == trackerTypes[0]:
		    tracker = cv2.TrackerBoosting_create()
		elif self.trackerType == trackerTypes[1]: 
		    tracker = cv2.TrackerMIL_create()
		elif self.trackerType == trackerTypes[2]:
		    tracker = cv2.TrackerKCF_create()
		elif self.trackerType == trackerTypes[3]:
		    tracker = cv2.TrackerTLD_create()
		elif self.trackerType == trackerTypes[4]:
		    tracker = cv2.TrackerMedianFlow_create()
		elif self.trackerType == trackerTypes[5]:
		    tracker = cv2.TrackerGOTURN_create()
		elif self.trackerType == trackerTypes[6]:
		    tracker = cv2.TrackerMOSSE_create()
		elif self.trackerType == trackerTypes[7]:
		    tracker = cv2.TrackerCSRT_create()
		else:
		    tracker = None
		    print('Incorrect tracker name')
		    print('Available trackers are:')
		    for t in trackerTypes:
		        print(t)
		 
		return tracker

	def update_tracker(self):
		#ROI FORMAT = x,y,w,h
		#Current format = y1, x1, y2, x2
		print(" Current Format"+str(settings.GLOBAL_BBOXES))
		settings.GLOBAL_TRACKED_BBOXES = []
		self.multiTracker = []
		image = settings.GLOBAL_FRAME.copy()
		for i, global_bbox_complex in enumerate(settings.GLOBAL_BBOXES):
			bbox_complex = global_bbox_complex.copy()
			bbox = bbox_complex['roi']
			bbox = int(bbox[1]), int(bbox[0]), abs(int(bbox[3]) - int(bbox[1])), abs(int(bbox[2]) - int(bbox[0]))
			bbox = tuple(bbox)
			tracker = self.createTrackerByName(self.trackerType)
			print("Tracked box init : "+str(bbox))
			success = tracker.init(image, bbox)
			if success:
				element_tracker = {'tracker': tracker, 'bbox': bbox_complex}
				self.multiTracker.append(element_tracker)
			settings.GLOBAL_TRACKED_BBOXES.append(bbox_complex)

	

	def intersection_over_union(self, box_a, box_b):
		# Determine the coordinates of each of the two boxes
		xA = max(box_a[0], box_b[0])
		yA = max(box_a[1], box_b[1])
		xB = min(box_a[0]+box_a[2], box_b[0]+box_b[2])
		yB = min(box_a[1]+box_a[3], box_b[1]+box_b[3])
		  
		# Calculate the area of the intersection area
		area_of_intersection = (xB - xA + 1) * (yB - yA + 1)
		 
		# Calculate the area of both rectangles
		box_a_area = (box_a[2] + 1) * (box_a[3] + 1)
		box_b_area = (box_b[2] + 1) * (box_b[3] + 1)

		# Calculate the area of intersection divided by the area of union
		# Area of union = sum both areas less the area of intersection
		iou = area_of_intersection / float(box_a_area + box_b_area - area_of_intersection)
		
		# Return the score
		return iou

	def convert_tracker_result(self, tracker_list):
		return_list = []
		for tracker in tracker_list:
			bbox = (int(tracker['bbox']['roi'][1]), int(tracker['bbox']['roi'][0]), int(tracker['bbox']['roi'][3]+tracker['bbox']['roi'][1]), int(tracker['bbox']['roi'][2] + tracker['bbox']['roi'][0]) )		
			
			return_list.append({'roi' : bbox, 'mask' : tracker['bbox']['mask'], 'class_id' : tracker['bbox']['class_id'], 'score': tracker['bbox']['score']})
		return return_list

	def run(self):
		success_tracker = True
		old_bboxes = settings.GLOBAL_BBOXES

		while not settings.GLOBAL_SHUTDOWN_FLAG.is_set():
			#time.sleep(settings.GLOBAL_FRAMERATE)
			
			image = settings.GLOBAL_FRAME.copy()
			tracked_boxes = []
			for i,tracker in enumerate(self.multiTracker):
				success, tracked_box = self.multiTracker[i]['tracker'].update(image)
				if not success :
					print("Removing a tracker ...")
					self.multiTracker.pop(i)
				else:		
					self.multiTracker[i]['bbox']['roi'] = tracked_box
			settings.GLOBAL_TRACKED_BBOXES = self.convert_tracker_result(self.multiTracker)


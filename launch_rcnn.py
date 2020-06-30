#!/usr/bin/env python2.7
"""
MultiThreaded Mask R-CNN
Mask R-CNN model launch file.

Copyright (c) 2019.
Licensed under the MIT License (see LICENSE for details)
Written by Hugo JOBY
"""

from prediction import PredictionManager
from tracking import TrackerManager
from deeplearning import DeeplearningManager
from videoreader import Videoreader
import settings

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

import random, time
import threading
import signal



def add_bounding_boxes(frame, class_names, boxes, color = (250,0,0)):
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i, newbox in enumerate(boxes):
		p1 = (int(newbox['roi'][1]), int(newbox['roi'][0]))
		p2 = (int(newbox['roi'][3]), int(newbox['roi'][2]))
		cv2.rectangle(frame, p1, p2, color, 2, 1)
		text_to_print = str(class_names[newbox['class_id']])+" "+str(newbox['score'])+"%"
		cv2.putText(frame, text_to_print, p1, font, 0.7,(255,255,255),2,cv2.LINE_AA)
	return frame

def add_arrow(frame, arrow, color = (250,0,0)):
	for i, newbox in enumerate(arrow):
		p1 = newbox[0]
		p2 = newbox[1]
		cv2.arrowedLine(frame, p1, p2, color, 5, 8, 0, 0.1)
	return frame

class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """
    pass

def service_shutdown(signum, frame):
	print('Caught signal %d' % signum)
	raise ServiceExit

class RCNN_launcher():

	def __init__(self, ):
		settings.init()

		#lock on bboxes

		self.frame = settings.GLOBAL_FRAME 
		self.bboxes = settings.GLOBAL_BBOXES
		self.predicted = settings.GLOBAL_PREDICTED_BBOXES
		self.tracked = settings.GLOBAL_TRACKED_BBOXES
		self.lock = settings.GLOBAL_THREAD_LOCK
		
		self._vidcap = Videoreader(self.lock, settings.GLOBAL_FRAME , 1, "output.mp4")
		self._prediction = PredictionManager(self.frame, self.bboxes, self.lock)
		self._tracking = TrackerManager(self.frame, self.bboxes, self.lock)
		self._deeplearning = DeeplearningManager(self.frame, self.bboxes, self.lock)
		
		#thread list
		self.threads = [self._vidcap, self._tracking, self._deeplearning, self._tracking]

		self.windowname = "Input Frame"
		

		

	def draw_frame(self):
		"""Used to draw the bouding box on the screen"""
		#drawed_frame = add_bounding_boxes(settings.GLOBAL_FRAME, settings.GLOBAL_CLASS_NAME, settings.GLOBAL_BBOXES)
		detect_bboxes = settings.GLOBAL_BBOXES
		tracked_bboxes = settings.GLOBAL_TRACKED_BBOXES
		predicted_bboxes = settings.GLOBAL_PREDICTED_BBOXES

		drawed_frame = settings.GLOBAL_FRAME.copy()
		#drawed_frame = add_bounding_boxes(drawed_frame, settings.GLOBAL_CLASS_NAME, detect_bboxes)
		drawed_frame = add_bounding_boxes(drawed_frame, settings.GLOBAL_CLASS_NAME, tracked_bboxes, (0,255,0))
		#drawed_frame = add_bounding_boxes(drawed_frame, settings.GLOBAL_CLASS_NAME, predicted_bboxes, (0,0,255))
		
		drawed_frame = add_arrow(drawed_frame, settings.GLOBAL_PREDICTED_TRAJECTORY, (0,0,255))
		cv2.imshow(self.windowname, np.array(drawed_frame, dtype = np.uint8 ))

	def execute(self):

		# Register the signal handlers
		signal.signal(signal.SIGTERM, service_shutdown)
		signal.signal(signal.SIGINT, service_shutdown)

		
		#initialize the first frame
		settings.GLOBAL_FRAME  = self._vidcap.getSingleframe()

		#initialize bboxes
		self._deeplearning.return_image_rcnn(settings.GLOBAL_FRAME)

		#initialize tracker
		self._tracking.update_tracker()

		#initialize prediction 
		self._prediction.get_predicted_trajectory()

		self.draw_frame()

		try:
			self._vidcap.start()
			self._deeplearning.start()
			self._tracking.start()
			self._prediction.start()

			while not settings.GLOBAL_SHUTDOWN_FLAG.is_set():
				self.draw_frame()

				k = cv2.waitKey(1)
				if k == 27 : 
					raise ServiceExit
					break # esc pressed
				elif k == 112 : cv2.waitKey(0) # p pressed for pause

	 
		except ServiceExit:
			print("The program is about to shutdown...")
			# Terminate the running threads.
			# Set the shutdown flag on each thread to trigger a clean shutdown of each thread.
			for thread in self.threads:
				"""Waits for the threads to csomplete before moving on
					with the main script.
				"""
				settings.GLOBAL_SHUTDOWN_FLAG.set()
				thread.join()

		for thread in self.threads:
			"""Waits for the threads to complete before moving on
				with the main script.
			"""
			thread.shutdown_flag.set()
			thread.join()

		cv2.destroyAllWindows()
		print('Exiting main program')


if __name__ == '__main__':
	_manager = RCNN_launcher()
	_manager.execute()

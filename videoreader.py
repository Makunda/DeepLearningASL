#!/usr/bin/env python2.7
"""
MultiThreaded Mask R-CNN
Mask R-CNN model launch file.

Copyright (c) 2019.
Licensed under the MIT License (see LICENSE for details)
Written by Hugo JOBY
"""

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
import settings
import signal



class Videoreader(threading.Thread):
	"""docstring for Videoreader"""
	    #Tracker parameters

	def __init__(self, lock_bboxes , frame = None,  mode = 1, video_name = "output.mp4"):
		threading.Thread.__init__(self)
		self.shutdown_flag = threading.Event()

		
		self.frame = settings.GLOBAL_FRAME
		self.old_frame = settings.GLOBAL_OLD_FRAME

		self.video_name = video_name
		self.mode = mode

		self.lock_bboxes = lock_bboxes
		# Root directory of the project
		self.ROOT_DIR = os.getcwd()

		# Directory of images to run detection on
		self.IMAGE_DIR = os.path.join(self.ROOT_DIR, "images")
		self.VIDEO_DIR = os.path.join(self.ROOT_DIR, "video")

		self.vidcap = self.videoReader()

	def getFrame(self):
		return self.frame

	def setFrame(self, frame):
		self.frame = frame


	def videoReader(self):
		#From Camera
		if self.mode == 1:
			try:
				_videoReader = cv2.VideoCapture(0)
			except IOError:
				print("Unable to open the webcam flux ...")
				sys.exit(-1)
		#From Video
		elif self.mode == 2:
			try:
				_videoReader = cv2.VideoCapture(self.VIDEO_DIR+'/'+str(self.video_name))
			except IOError:
				print("Unable to open the video flux ...")
				sys.exit(-1)
		else:
			_videoReader = None
			print('Incorrect Videoreader mode')
			sys.exit(0)
		return _videoReader

	def getSingleframe(self):
		#process the first frame
		while True:
			try :
				success,current_frame = self.vidcap.read()
				#print("Sucess ? "+str(success))
			except Exception as e:
				print("Error reading video flux ...")
				print e
				settings.GLOBAL_SHUTDOWN_FLAG.set()
			if not current_frame is None:
				break

		height, width, channels = current_frame.shape
		if height != 480 or width != 720:
			current_frame = cv2.resize(current_frame,(720,480))
		return current_frame

	def apply_treshold(self, image_curr, image_prec):
		diff = cv2.absdiff(image_prec, image_curr)
		mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

		th = 1
		imask =  mask>th

		canvas = np.zeros_like(image_curr, np.uint8)
		canvas[imask] = image_curr[imask]

		img_bw = 255*(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')

		se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
		se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
		mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

		mask = np.dstack([mask, mask, mask]) / 255
		out = canvas * mask
		cv2.imshow('Treshold vision Frame ', np.array(out, dtype = np.uint8 ))
		return out

	def background_substractor(self, image_curr):
		
		return return_image

	def run(self):
		print("Thread video Ready")
		self.getSingleframe()

		#pause for debug
		time.sleep(0.2)

		success_frame = True
		success_tracker = True
		while success_frame and not settings.GLOBAL_SHUTDOWN_FLAG.is_set():
			#Get time
			time.sleep(settings.GLOBAL_FRAMERATE)
			#Read a frame on the video
			settings.GLOBAL_OLD_FRAME = settings.GLOBAL_FRAME
			settings.GLOBAL_FRAME_LOCK.acquire()
			
			settings.GLOBAL_FRAME  = self.getSingleframe()
			settings.GLOBAL_FRAME_LOCK.release()
			


		print("Quit, Sucess : "+str(success_frame))
		#Realease the videoflux
		self.vidcap.release()
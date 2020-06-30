#!/usr/bin/env python2.7

import cv2 as cv
import numpy as np
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

import random, time
import threading
import settings
import signal


# Instantiate OCV kalman filter
class KalmanFilter:

    kf = cv.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    cv2.setIdentity(kf.measurementMatrix)
    cv2.setIdentity(kf.processNoiseCov, 1e-5)
    cv2.setIdentity(kf.measurementNoiseCov, 1e-1)
    cv2.setIdentity(kf.errorCovPost, 1)

    cv2.randn(kf.statePost, 0, 0.1)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        #print("PREDICTED ===== "+str(predicted))
        return predicted

class PredictionManager(threading.Thread):
    """docstring for PredictionManager"""
    def __init__(self, frame, bboxes, lock_bboxes):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()

        self.lock_bboxes = lock_bboxes
        self.bboxes = bboxes
        self.frame = frame
        self.newbboxes = []

    def DetectObject(self, pos_x, pos_y):
        # Create Kalman Filter Object
        kfObj = KalmanFilter()
        predictedCoords = np.zeros((2, 1), np.float32)
        predictedCoords = kfObj.Estimate(pos_x, pos_y)
        return predictedCoords

    def setFrame(self, frame):
        self.frame = frame

    def setBoxes(self, bboxes):
        self.bboxes = bboxes

    def getFrame(self):
        return self.frame

    def getBoxes(self):
        return self.bboxes

    def getNewboxes(self):
        return self.bboxes

    def convert_prediction_result(self, predicted_boxes):
        return_list = []
        for bbox in predicted_boxes:
            return_list.append({'roi' : bbox, 'mask' : None, 'class_id' : 1, 'score': None})
        return return_list



    def Predictbboxes(self, bboxes):
        newbboxes = []
        
        for bbox in bboxes:
            y, x, h, w = int(bbox['roi'][0]), int(bbox['roi'][1]), abs(int(bbox['roi'][2]) - int(bbox['roi'][0])), abs(int(bbox['roi'][3]) - int(bbox['roi'][1]))
            center_y, center_x = y+h/2, x+w/2 
            predicted_coord = self.DetectObject(center_x, center_y)
            new_center_x, new_center_y = predicted_coord[0], predicted_coord[1]
            newbox = [int(new_center_y - h/2), int(new_center_x - w/2), int((new_center_y + h/2)), int((new_center_x + w/2))]
            
            newbboxes.append(newbox)
        return newbboxes


    def get_predicted_trajectory(self):
        bbox_to_predict = settings.GLOBAL_TRACKED_BBOXES
        print("To predict : " + str(bbox_to_predict))
        old_prediction = settings.GLOBAL_TRACKED_BBOXES
        settings.GLOBAL_PREDICTED_BBOXES = []

        new_prediction = self.Predictbboxes(bbox_to_predict)

        #A corriger
        settings.GLOBAL_PREDICTED_BBOXES = self.convert_prediction_result(new_prediction)
        print("After : " + str( settings.GLOBAL_PREDICTED_BBOXES))
        trajectory_list = []
        for old, new in zip(old_prediction, new_prediction):
            y1, x1, h1, w1= int(old['roi'][1]), int(old['roi'][0]), abs(int(old['roi'][3]) - int(old['roi'][1])), abs(int(old['roi'][2]) - int(old['roi'][0]))
            y2, x2, h2, w2 = int(new[1]), int(new[0]), abs(int(new[3]) - int(new[1])), abs(int(new[2]) - int(new[0]))
            vector = [(y1+h1/2, x1+w1/2), (int((y2+h2/2)*1.1), int((x2+w2/2)*1.1))]
            trajectory_list.append(vector)

        settings.GLOBAL_PREDICTED_TRAJECTORY = trajectory_list
        return trajectory_list
 

    def run(self):
        success = 1
        while not settings.GLOBAL_SHUTDOWN_FLAG.is_set():
            #time.sleep(settings.GLOBAL_FRAMERATE)
            
            self.get_predicted_trajectory()

            

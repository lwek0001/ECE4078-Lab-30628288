# basic python packages
import numpy as np
import cv2 
import os, sys
import time

# import utility functions
#sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco 

from operate import Operate 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--yolo_model", default='YOLO/model/yolov8_model.pt')
    args, _ = parser.parse_known_args()
    operate = Operate(args)

while True: 
    operate.take_pic()
    cv2_imshow()
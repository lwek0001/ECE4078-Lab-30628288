# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import matplotlib.pyplot as plt

# import SLAM components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure

from AStarMod import AStarPlanner
show_animation = True
class navigation:
    def __init__(self, ekf, pibot):
        self.ekf = ekf
        self.pibot = pibot
        self.parser = argparse.ArgumentParser("Fruit searching")
        self.parser.add_argument("--map", type=str, default='TrueMap.txt') # change to 'M4_true_map_part.txt' for lv2&3
        self.parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
        self.parser.add_argument("--port", metavar='', type=int, default=8080)
        self.parser.add_argument("--calib_dir", type=str, default="calibration/param/")
        self.args, _ = self.parser.parse_known_args()
        self.datadir = self.args.calib_dir
        self.ip = self.args.ip
        self.fileK = "{}intrinsic.txt".format(self.datadir)
        self.fileD = "{}distCoeffs.txt".format(self.datadir)
        self.fileS = "{}scale.txt".format(self.datadir)
        self.scale = np.loadtxt(self.fileS, delimiter=',')
        if self.ip == 'localhost':
            scale /= 2
        self.fileB = "{}baseline.txt".format(self.datadir)
        self.baseline = np.loadtxt(self.fileB, delimiter=',')
        
    def read_true_map(self, fname):
        """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

        @param fname: filename of the map
        @return:
            1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
            2) locations of the targets, [[x1, y1], ..... [xn, yn]]
            3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
        """
        with open(fname, 'r') as fd:
            gt_dict = json.load(fd)
            fruit_list = []
            fruit_true_pos = []
            aruco_true_pos = np.empty([10, 2])

            # remove unique id of targets of the same type
            for key in gt_dict:
                x = np.round(gt_dict[key]['x'], 2)
                y = np.round(gt_dict[key]['y'], 2)

                if key.startswith('aruco'):
                    if key.startswith('aruco10'):
                        aruco_true_pos[9][0] = x
                        aruco_true_pos[9][1] = y
                    else:
                        marker_id = int(key[5]) - 1
                        aruco_true_pos[marker_id][0] = x
                        aruco_true_pos[marker_id][1] = y
                else:
                    fruit_list.append(key[:-2])
                    if len(fruit_true_pos) == 0:
                        fruit_true_pos = np.array([[x, y]])
                    else:
                        fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

            return fruit_list, fruit_true_pos, aruco_true_pos

    def read_search_list(self):
        """Read the search order of the target fruits

        @return: search order of the target fruits
        """
        search_list = []
        with open('shopping_list.txt', 'r') as fd:
            fruits = fd.readlines()

            for fruit in fruits:
                search_list.append(fruit.strip())

        return search_list

    def print_target_fruits_pos(self, search_list, fruit_list, fruit_true_pos):
        """Print out the target fruits' pos in the search order

        @param search_list: search order of the fruits
        @param fruit_list: list of target fruits
        @param fruit_true_pos: positions of the target fruits
        """
        fruit_search_list_dict = dict()
        print("Search order:")
        n_fruit = 1

        for fruit in search_list:
            counter = 1
            for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
                if fruit == fruit_list[i]:
                    #old_n_fruit += 1
                    
                    print('{}) {} at [{}, {}]'.format(n_fruit,
                                                    fruit,
                                                    np.round(fruit_true_pos[i][0], 2),
                                                    np.round(fruit_true_pos[i][1], 2)))
                    
                    list = [np.round(fruit_true_pos[i][0], 2), np.round(fruit_true_pos[i][1], 2)]
                    
                    n_fruit += 1
                    if counter == 1:
                        fruit_search_list_dict[fruit] = list
                    else: 
                        fruit_search_list_dict[fruit].append(list[0])
                        fruit_search_list_dict[fruit].append(list[1])
                    counter += 1
            # n_fruit += 1
        
        return fruit_search_list_dict

    def get_robot_pose(self):
        robot_pose = self.ekf.get_state_vector()
        return robot_pose

    def manual_movement(self):
        # The following is only a skeleton code for semi-auto navigation
        while True:
            # enter the waypoints
            # instead of manually enter waypoints, you can give coordinates by clicking on a map, see camera_calibration.py from M2
            x,y = 0.0,0.0
            x = input("X coordinate of the waypoint: ")
            try:
                x = float(x)
            except ValueError:
                print("Please enter a number.")
                continue
            y = input("Y coordinate of the waypoint: ")
            try:
                y = float(y)
            except ValueError:
                print("Please enter a number.")
                continue

            # estimate the robot's pose
            robot_pose = self.get_robot_pose()

            # robot drives to the waypoint
            waypoint = [x,y]
            new_pose_angle = self.drive_to_point(waypoint,robot_pose)
            updated_pose = np.array([x,y,new_pose_angle]).reshape((3,1))
            
            robot_pose = self.get_robot_pose()
            print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

            # exit
            self.pibot.set_velocity([0, 0])
            uInput = input("Add a new waypoint? [Y/N]")
            if uInput == 'N' or uInput == 'n':
                break

    def rotation(self, waypoint):
        wheel_vel = 30 # tick
        time_revolution = np.pi*self.baseline/(wheel_vel*self.scale)
        # turn towards the waypoint
        desired_pose = np.arctan2(waypoint[1]-float(self.get_robot_pose()[1][0]),waypoint[0]-float(self.get_robot_pose()[0][0]))
        new_pose_angle = desired_pose-float(self.get_robot_pose()[2][0])
        new_pose_angle = np.round(np.arctan2(np.sin(new_pose_angle),np.cos(new_pose_angle)),3)
        
        turn_time = np.round(abs((new_pose_angle)/(2*np.pi)*time_revolution),3)
        print("Rotating {:.2f} radians".format(new_pose_angle))
        print("Turning for {:.2f} seconds".format(turn_time))
        
        if new_pose_angle > 0:
            lv, rv = self.pibot.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)    
        elif new_pose_angle < 0:
            lv, rv = self.pibot.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
        else: 
            lv, rv = self.pibot.set_velocity([0, 0], turning_tick=wheel_vel, time=turn_time)
        
        return lv, rv, turn_time
    
    def linear_movement(self, waypoint):
        # after turning, drive straight to the waypoint
        euclidean_distance = np.sqrt((waypoint[0]-self.get_robot_pose()[0][0])**2 + (waypoint[1]-self.get_robot_pose()[1][0])**2)
        wheel_vel = 50 # tick 
        time_metre = 1/(wheel_vel*self.scale)
        drive_time = np.round(abs(euclidean_distance * time_metre),3)
        print("Driving for {:.2f} seconds".format(drive_time))
        lv, rv = self.pibot.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
        return lv, rv, drive_time
    
        
    
    def path_planning(self, fruit):
        sx = self.get_robot_pose()[0][0]
        sy = self.get_robot_pose()[1][0]
        goal = self.search_list_dict[fruit]

    
        ox = self.ox
        oy = self.oy

        for j in range(-self.fruit_size,self.fruit_size):
            np.delete(ox, np.where(self.search_list_dict[fruit][0]*100+self.fruit_size))
            np.delete(oy, np.where(self.search_list_dict[fruit][1]*100+j))
        for j in range(-self.fruit_size,self.fruit_size):
            np.delete(ox, np.where(self.search_list_dict[fruit][0]*100-self.fruit_size))
            np.delete(oy, np.where(self.search_list_dict[fruit][1]*100+j))
        for j in range(-self.fruit_size,self.fruit_size+1):
            np.delete(oy, np.where(self.search_list_dict[fruit][0]*100+self.fruit_size))
            np.delete(ox, np.where(self.search_list_dict[fruit][1]*100+j))
        for j in range(-self.fruit_size,self.fruit_size+1):
            np.delete(oy, np.where(self.search_list_dict[fruit][0]*100-self.fruit_size))
            np.delete(ox, np.where(self.search_list_dict[fruit][1]*100+j))

        if show_animation:  # pragma: no cover
            plt.plot(self.boundaries_x, self.boundaries_y, ".k")
            plt.plot(self.arucos_x, self.arucos_y, "bs")
            plt.plot(self.obs_fruit_x, self.obs_fruit_y, "ro")
            plt.plot(self.non_obs_fruit_x, self.non_obs_fruit_y, "go")
            plt.grid(True)
            plt.axis("equal")
            
        a_star = AStarPlanner(ox, oy, 15, self.robot_radius, True)
        rx, ry = a_star.planning(sx*100.0, sy*100.0, goal[0]*100.0, goal[1]*100.0, self.radius_threshold)
        
        rx = rx[::-1]
        ry = ry[::-1]

        if show_animation:  # pragma: no cover
            plt.axis("equal")
            plt.plot(rx[0:2], ry[0:2], "-r")
            plt.pause(0.001)
            plt.show(block = False)
            plt.pause(0.001)

        if len(rx) == 1:
            x_m = rx[0]/100
            y_m = ry[0]/100 
            waypoint = [x_m, y_m]
            drive_flag = False
        else:
            x_m = rx[1]/100
            y_m = ry[1]/100 
            waypoint = [x_m, y_m]
            drive_flag = True
        return waypoint, drive_flag

    def map_generation(self, grid_size, boundary_size): 
        self.boundaries_x, self.boundaries_y = [], []

        #Arena Boundaries 
        for i in range(-boundary_size, boundary_size):
            self.boundaries_x.append(i)
            self.boundaries_y.append(-boundary_size)
        for i in range(-boundary_size, boundary_size):
            self.boundaries_x.append(boundary_size)
            self.boundaries_y.append(i)
        for i in range(-boundary_size, boundary_size):
            self.boundaries_x.append(i)
            self.boundaries_y.append(boundary_size)
        for i in range(-boundary_size, boundary_size):
            self.boundaries_x.append(-boundary_size)
            self.boundaries_y.append(i)

        self.arucos_x, self.arucos_y = [], []
        # Aruco Markers 
        aruco_true_pos = self.read_true_map('TrueMap.txt')[2]

        for i in range(len(aruco_true_pos)):
            for j in range(-self.marker_size,self.marker_size):
                self.arucos_x.append(aruco_true_pos[i][0]*100+self.marker_size)
                self.arucos_y.append(aruco_true_pos[i][1]*100+j)
            for j in range(-self.marker_size,self.marker_size):
                self.arucos_x.append(aruco_true_pos[i][0]*100-self.marker_size)
                self.arucos_y.append(aruco_true_pos[i][1]*100+j)
            for j in range(-self.marker_size,self.marker_size+1):
                self.arucos_x.append(aruco_true_pos[i][0]*100+j)
                self.arucos_y.append(aruco_true_pos[i][1]*100+self.marker_size)
            for j in range(-self.marker_size,self.marker_size+1):
                self.arucos_x.append(aruco_true_pos[i][0]*100+j)
                self.arucos_y.append(aruco_true_pos[i][1]*100-self.marker_size)
        
        self.obs_fruit_x, self.obs_fruit_y = [], []
        # Obstacle Fruits
        for i in self.obstacle_list_dict.keys():
            for j in range(-self.fruit_size,self.fruit_size):
                self.obs_fruit_x.append(self.obstacle_list_dict[i][0]*100+self.fruit_size)
                self.obs_fruit_y.append(self.obstacle_list_dict[i][1]*100+j)
                if len(self.obstacle_list_dict[i]) > 2:
                    self.obs_fruit_x.append(self.obstacle_list_dict[i][2]*100+self.fruit_size)
                    self.obs_fruit_y.append(self.obstacle_list_dict[i][2+1]*100+j)
            for j in range(-self.fruit_size,self.fruit_size):
                self.obs_fruit_x.append(self.obstacle_list_dict[i][0]*100-self.fruit_size)
                self.obs_fruit_y.append(self.obstacle_list_dict[i][1]*100+j)
                if len(self.obstacle_list_dict[i]) > 2:
                    
                    self.obs_fruit_x.append(self.obstacle_list_dict[i][2]*100-self.fruit_size)
                    self.obs_fruit_y.append(self.obstacle_list_dict[i][2+1]*100+j)
            for j in range(-self.fruit_size,self.fruit_size+1):
                self.obs_fruit_x.append(self.obstacle_list_dict[i][0]*100+j)
                self.obs_fruit_y.append(self.obstacle_list_dict[i][1]*100+self.fruit_size)
                if len(self.obstacle_list_dict[i]) > 2:
                    
                    self.obs_fruit_x.append(self.obstacle_list_dict[i][2]*100+j)
                    self.obs_fruit_y.append(self.obstacle_list_dict[i][2+1]*100+self.fruit_size)
            for j in range(-self.fruit_size,self.fruit_size+1):
                self.obs_fruit_x.append(self.obstacle_list_dict[i][0]*100+j)
                self.obs_fruit_y.append(self.obstacle_list_dict[i][1]*100-self.fruit_size)
                if len(self.obstacle_list_dict[i]) > 2:
                
                    self.obs_fruit_x.append(self.obstacle_list_dict[i][2]*100+j)
                    self.obs_fruit_y.append(self.obstacle_list_dict[i][2+1]*100-self.fruit_size)
        
        self.non_obs_fruit_x, self.non_obs_fruit_y = [], []
        # Non-searching Fruits 
        for i in self.search_list_dict.keys():
            for j in range(-self.fruit_size,self.fruit_size):
                self.non_obs_fruit_x.append(self.search_list_dict[i][0]*100+self.fruit_size)
                self.non_obs_fruit_y.append(self.search_list_dict[i][1]*100+j)
            for j in range(-self.fruit_size,self.fruit_size):
                self.non_obs_fruit_x.append(self.search_list_dict[i][0]*100-self.fruit_size)
                self.non_obs_fruit_y.append(self.search_list_dict[i][1]*100+j)
            for j in range(-self.fruit_size,self.fruit_size+1):
                self.non_obs_fruit_x.append(self.search_list_dict[i][0]*100+j)
                self.non_obs_fruit_y.append(self.search_list_dict[i][1]*100+self.fruit_size)
            for j in range(-self.fruit_size,self.fruit_size+1):
                self.non_obs_fruit_x.append(self.search_list_dict[i][0]*100+j)
                self.non_obs_fruit_y.append(self.search_list_dict[i][1]*100-self.fruit_size)

        ox, oy = [], []
        ox.append(self.boundaries_x)
        ox.append(self.arucos_x)
        ox.append(self.obs_fruit_x)
        ox.append(self.non_obs_fruit_x)
        oy.append(self.boundaries_y)
        oy.append(self.arucos_y)
        oy.append(self.obs_fruit_y)
        oy.append(self.non_obs_fruit_y)
        self.ox = np.concatenate(ox)
        self.oy = np.concatenate(oy)
        self.a_star = AStarPlanner(self.ox, self.oy, grid_size, self.robot_radius, False)

    def inputs(self):
        self.fruits_list, self.fruits_true_pos, self.aruco_true_pos = self.read_true_map(self.args.map)
        self.search_list = self.read_search_list()
        self.search_list_dict = self.print_target_fruits_pos(self.search_list, self.fruits_list, self.fruits_true_pos)
        self.obstacle_list = list(set(self.fruits_list)-set(self.search_list))
        self.obstacle_list_dict = self.print_target_fruits_pos(self.obstacle_list, self.fruits_list, self.fruits_true_pos)

        self.driving_option, self.marker_size, self.fruit_size, self.radius_threshold, self.robot_radius = input("manual or automatic drive? [M/A] ,marker threshold?, fruit_threshold?, radius_threshold? , robot_radius?").split(", ",5)
        self.marker_size = int(self.marker_size)
        self.fruit_size = int(self.fruit_size)
        self.radius_threshold = int(self.radius_threshold)
        self.robot_radius = int(self.robot_radius)

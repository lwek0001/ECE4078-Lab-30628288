# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import Path Planning 
from AStarMod import AStarPlanner

# import utility functions
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure

show_animation = True

target_dimensions_dict = {'orange': [0.08,0.08,0.075], 'lemon': [0.055,0.055,0.075], 
                              'lime': [0.055,0.055,0.075], 'tomato': [0.070,0.070,0.060], 
                              'capsicum': [0.080,0.080,0.100], 'potato': [0.100,0.070,0.070], 
                              'pumpkin': [0.090,0.090,0.100], 'garlic': [0.060,0.060,0.080]}

def read_true_map(fname):
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


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('shopping_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
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


def get_robot_pose():
    robot_pose = EKF_slam.get_state_vector()
    return robot_pose


def set_robot_pose(updated_pose):
    EKF_slam.set_state_vector(updated_pose)

def check_eDist(pose, waypoint, threshold):
    pose=np.array(pose).reshape(2,1)
    waypoint=np.array(waypoint).reshape(2,1)
    eDist = np.linalg.norm(pose-waypoint)
    if eDist < threshold:
        return True
    else:
        return False

def manual_movement():
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
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        new_pose_angle = drive_to_point(waypoint,robot_pose)
        updated_pose = np.array([x,y,new_pose_angle]).reshape((3,1))
        set_robot_pose(updated_pose)
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N' or uInput == 'n':
            break


def drive_to_point(waypoint, robot_pose, drive_flag = True):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    wheel_vel = 30 # tick
    time_revolution = np.pi*baseline/(wheel_vel*scale)
    # turn towards the waypoint
    desired_pose = np.arctan2(waypoint[1]-float(robot_pose[1,0]),waypoint[0]-float(robot_pose[0,0]))
    new_pose_angle = desired_pose-float(robot_pose[2,0])
    new_pose_angle = np.arctan2(np.sin(new_pose_angle),np.cos(new_pose_angle))
    
    turn_time = abs((new_pose_angle)/(2*np.pi)*time_revolution)
    print("Rotating {:.2f} radians".format(new_pose_angle))
    print("Turning for {:.2f} seconds".format(turn_time))
    if drive_flag:
        if new_pose_angle > 0:
            ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)    
        elif new_pose_angle < 0:
            ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
    
    
    # after turning, drive straight to the waypoint
    euclidean_distance = np.sqrt((waypoint[0]-robot_pose[0,0])**2 + (waypoint[1]-robot_pose[1,0])**2)
    wheel_vel = 50 # tick 
    time_metre = 1/(wheel_vel*scale)
    drive_time = abs(euclidean_distance * time_metre)
    print("Driving for {:.2f} seconds".format(drive_time))
    if drive_flag:
        ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    orientation = np.mod(desired_pose, 2*np.pi)

    return(orientation)
    

def automatic_movement(search_list, search_list_dict, drive_flag, marker_size=3, fruit_size=5,  collision_threshold=0.35):
    print("Starting to search for fruits in 3 seconds...")
    time.sleep(3)
    for i in search_list: 
        coords_search = search_list_dict[i]
        print("Fruit: {}, Location: {}".format(i, coords_search))
        robot_pose = get_robot_pose()
        rx, ry = find_path(float(robot_pose[0])*100.0, float(robot_pose[1])*100.0, coords_search[0]*100.0, coords_search[1]*100.0, 10, 15, 150, i, marker_size, fruit_size)
        # Reverse list and convert to m 
        print("Path found, driving to waypoint...")
        time.sleep(3)
        rx = rx[::-1]
        ry = ry[::-1]
        goal = [rx[-1]/100.0, ry[-1]/100.0]
        for i in range(len(rx)):
            if i != len(rx)-1:
                robot_pose = get_robot_pose()
                x_m = rx[i+1]/100
                y_m = ry[i+1]/100 
                waypoint = [x_m, y_m]
                new_pose_angle = drive_to_point(waypoint,robot_pose, drive_flag=drive_flag)
                updated_pose = np.array([x_m,y_m,new_pose_angle]).reshape((3,1))
                set_robot_pose(updated_pose)
                if check_eDist(waypoint, goal, collision_threshold):
                    break
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        obstacle_list_dict[i] = coords_search
        if drive_flag:
            ppi.set_velocity([0, 0])    

def find_path(sx, sy, gx, gy, grid_size, robot_radius, boundary_size, fruit_search, marker_size, fruit_size): 
    print("Finding Path")
    boundaries_x, boundaries_y = [], []
    #Arena Boundaries 
    for i in range(-boundary_size, boundary_size):
        boundaries_x.append(i)
        boundaries_y.append(-boundary_size)
    for i in range(-boundary_size, boundary_size):
        boundaries_x.append(boundary_size)
        boundaries_y.append(i)
    for i in range(-boundary_size, boundary_size):
        boundaries_x.append(i)
        boundaries_y.append(boundary_size)
    for i in range(-boundary_size, boundary_size):
        boundaries_x.append(-boundary_size)
        boundaries_y.append(i)

    arucos_x, arucos_y = [], []
    # Aruco Markers 
    aruco_true_pos = read_true_map('map_full.txt')[2]

    for i in range(len(aruco_true_pos)):
        for j in range(-marker_size,marker_size):
            arucos_x.append(aruco_true_pos[i][0]*100+marker_size)
            arucos_y.append(aruco_true_pos[i][1]*100+j)
        for j in range(-marker_size,marker_size):
            arucos_x.append(aruco_true_pos[i][0]*100-marker_size)
            arucos_y.append(aruco_true_pos[i][1]*100+j)
        for j in range(-marker_size,marker_size+1):
            arucos_x.append(aruco_true_pos[i][0]*100+j)
            arucos_y.append(aruco_true_pos[i][1]*100+marker_size)
        for j in range(-marker_size,marker_size+1):
            arucos_x.append(aruco_true_pos[i][0]*100+j)
            arucos_y.append(aruco_true_pos[i][1]*100-marker_size)
    
    obs_fruit_x, obs_fruit_y = [], []
    # Obstacle Fruits
    for i in obstacle_list_dict.keys():
        for j in range(-fruit_size,fruit_size):
            obs_fruit_x.append(obstacle_list_dict[i][0]*100+fruit_size)
            obs_fruit_y.append(obstacle_list_dict[i][1]*100+j)
            if len(obstacle_list_dict[i]) > 2:
                obs_fruit_x.append(obstacle_list_dict[i][2]*100+fruit_size)
                obs_fruit_y.append(obstacle_list_dict[i][2+1]*100+j)
        for j in range(-fruit_size,fruit_size):
            obs_fruit_x.append(obstacle_list_dict[i][0]*100-fruit_size)
            obs_fruit_y.append(obstacle_list_dict[i][1]*100+j)
            if len(obstacle_list_dict[i]) > 2:
                
                obs_fruit_x.append(obstacle_list_dict[i][2]*100-fruit_size)
                obs_fruit_y.append(obstacle_list_dict[i][2+1]*100+j)
        for j in range(-fruit_size,fruit_size+1):
            obs_fruit_x.append(obstacle_list_dict[i][0]*100+j)
            obs_fruit_y.append(obstacle_list_dict[i][1]*100+fruit_size)
            if len(obstacle_list_dict[i]) > 2:
                
                obs_fruit_x.append(obstacle_list_dict[i][2]*100+j)
                obs_fruit_y.append(obstacle_list_dict[i][2+1]*100+fruit_size)
        for j in range(-fruit_size,fruit_size+1):
            obs_fruit_x.append(obstacle_list_dict[i][0]*100+j)
            obs_fruit_y.append(obstacle_list_dict[i][1]*100-fruit_size)
            if len(obstacle_list_dict[i]) > 2:
               
                obs_fruit_x.append(obstacle_list_dict[i][2]*100+j)
                obs_fruit_y.append(obstacle_list_dict[i][2+1]*100-fruit_size)
    
    non_obs_fruit_x, non_obs_fruit_y = [], []
    # Non-searching Fruits 
    for i in search_list_dict.keys():
        if i != fruit_search:
            for j in range(-fruit_size,fruit_size):
                non_obs_fruit_x.append(search_list_dict[i][0]*100+fruit_size)
                non_obs_fruit_y.append(search_list_dict[i][1]*100+j)
            for j in range(-fruit_size,fruit_size):
                non_obs_fruit_x.append(search_list_dict[i][0]*100-fruit_size)
                non_obs_fruit_y.append(search_list_dict[i][1]*100+j)
            for j in range(-fruit_size,fruit_size+1):
                non_obs_fruit_x.append(search_list_dict[i][0]*100+j)
                non_obs_fruit_y.append(search_list_dict[i][1]*100+fruit_size)
            for j in range(-fruit_size,fruit_size+1):
                non_obs_fruit_x.append(search_list_dict[i][0]*100+j)
                non_obs_fruit_y.append(search_list_dict[i][1]*100-fruit_size)

    if show_animation:  # pragma: no cover
        plt.plot(boundaries_x, boundaries_y, ".k")
        plt.plot(arucos_x, arucos_y, "bs")
        plt.plot(obs_fruit_x, obs_fruit_y, "ro")
        plt.plot(non_obs_fruit_x, non_obs_fruit_y, "go")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
        
    ox, oy = [], []
    ox.append(boundaries_x)
    ox.append(arucos_x)
    ox.append(obs_fruit_x)
    ox.append(non_obs_fruit_x)
    oy.append(boundaries_y)
    oy.append(arucos_y)
    oy.append(obs_fruit_y)
    oy.append(non_obs_fruit_y)
    ox = np.concatenate(ox)
    oy = np.concatenate(oy)
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show(block = False)
        plt.pause(0.001)
        
        
    return rx, ry

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='EstimateMap.txt') # change to 'M4_true_map_part.txt' for lv2&3
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    args, _ = parser.parse_known_args()
    datadir = args.calib_dir
    ip = args.ip
    fileK = "{}intrinsic.txt".format(datadir)
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileD = "{}distCoeffs.txt".format(datadir)
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    fileS = "{}scale.txt".format(datadir)
    scale = np.loadtxt(fileS, delimiter=',')
    if ip == 'localhost':
        scale /= 2
    fileB = "{}baseline.txt".format(datadir)
    baseline = np.loadtxt(fileB, delimiter=',')
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    EKF_slam = EKF(robot)

    ppi = PenguinPi(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    search_list_dict = print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    #obstacle_list = [x for x in fruits_list if not x in search_list or search_list.remove(x)]
    obstacle_list = list(set(fruits_list)-set(search_list))
    obstacle_list_dict = print_target_fruits_pos(obstacle_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    driving_option = input("manual or automatic drive? [M/A]: ")
    marker_threshold = input("marker threshold: ")
    fruit_threshold = input("fruit threshold: ")
    collision_threshold = input("collision threshold: ")
    

   
    if driving_option == 'M' or driving_option == 'm':
        manual_movement()
    elif driving_option == 'A' or driving_option == 'a':
        automatic_movement(search_list, search_list_dict, drive_flag=False, marker_size=int(marker_threshold), fruit_size=int(fruit_threshold), collision_threshold = float(collision_threshold))
    else:
        print("Please enter 'M' or 'A'.")

    plt.show()


    

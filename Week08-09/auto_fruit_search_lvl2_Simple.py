# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure


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
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

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
    with open('M4_prac_shopping_list.txt', 'r') as fd:
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
        for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
                fruit_search_list_dict[fruit] = [np.round(fruit_true_pos[i][0], 1), np.round(fruit_true_pos[i][1], 1)]
        n_fruit += 1
    
    return fruit_search_list_dict

# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# note that this function requires your camera and wheel calibration parameters from M2, and the "util" folder from M1
# fully automatic navigation:
# try developing a path-finding algorithm that produces the waypoints automatically
def drive_to_point(waypoint, robot_pose):
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
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    orientation = np.mod(desired_pose, 2*np.pi)

    return(orientation)
    
def get_robot_pose():
    robot_pose = EKF_slam.get_state_vector()
    return robot_pose

def set_robot_pose(updated_pose):
    EKF_slam.set_state_vector(updated_pose)

def automatic_movement(search_list, search_list_dict):
    print("Starting to search for fruits in 3 seconds...")
    time.sleep(3)
    for i in search_list: 
        coords_search = search_list_dict[i]
        print("Fruit: {} Location: {}".format(i, coords_search))
        robot_pose = get_robot_pose()
        waypoint = coords_search
        x = waypoint[0]
        y = waypoint[1]
        new_pose_angle = drive_to_point(waypoint,robot_pose)
        updated_pose = np.array([x,y,new_pose_angle]).reshape((3,1))
        set_robot_pose(updated_pose)
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        ppi.set_velocity([0, 0])
        time.sleep(3)
    print("Finished searching for all fruits, returning home...")
    waypoint = [0,0]
    new_pose_angle = drive_to_point(waypoint,robot_pose)
    updated_pose = np.array([x,y,new_pose_angle]).reshape((3,1))
    set_robot_pose(updated_pose)
    robot_pose = get_robot_pose()
    print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

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

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_prac_map_full.txt') # change to 'M4_true_map_part.txt' for lv2&3
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

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    driving_option = input("Do you want to drive the robot manually or automatically? [M/A]")
    if driving_option == 'M' or driving_option == 'm':
        manual_movement()
    elif driving_option == 'A' or driving_option == 'a':
        automatic_movement(search_list, search_list_dict)
    else:
        print("Please enter 'M' or 'A'.")


    

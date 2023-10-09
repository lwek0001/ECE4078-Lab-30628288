# estimate the pose of target objects detected
import numpy as np
import json
import os
import ast
import cv2
from YOLO.detector import Detector
from sklearn.metrics import silhouette_score 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# list of target fruits and vegs types
# Make sure the names are the same as the ones used in your YOLO model
TARGET_TYPES = ['orange', 'lemon', 'lime', 'tomato', 'capsicum', 'potato', 'pumpkin', 'garlic']

def fruit_input():
    fruit_count = {}

    for i in range(len(TARGET_TYPES)):
        fruit_count[TARGET_TYPES[i]] = (int(input(f'How many {TARGET_TYPES[i]}s are there? ')))
    return fruit_count

def estimate_pose(camera_matrix, obj_info, robot_pose):
    """
    function:
        estimate the pose of a target based on size and location of its bounding box and the corresponding robot pose
    input:
        camera_matrix: list, the intrinsic matrix computed from camera calibration (read from 'param/intrinsic.txt')
            |f_x, s,   c_x|
            |0,   f_y, c_y|
            |0,   0,   1  |
            (f_x, f_y): focal length in pixels
            (c_x, c_y): optical centre in pixels
            s: skew coefficient (should be 0 for PenguinPi)
        obj_info: list, an individual bounding box in an image (generated by get_bounding_box, [label,[x,y,width,height]])
        robot_pose: list, pose of robot corresponding to the image (read from 'lab_output/images.txt', [x,y,theta])
    output:
        target_pose: dict, prediction of target pose
    """
    # read in camera matrix (from camera calibration results)
    focal_length = camera_matrix[0][0]

    # there are 8 possible types of fruits and vegs
    ######### Replace with your codes #########
    
    # target_dimensions_dict = {'orange': [0.08,0.08,0.075], 'lemon': [0.055,0.055,0.075], 
    #                           'lime': [0.055,0.055,0.075], 'tomato': [0.070,0.070,0.060], 
    #                           'capsicum': [0.080,0.080,0.0100], 'potato': [0.0100,0.070,0.070], 
    #                           'pumpkin': [0.090,0.090,0.0100], 'garlic': [0.060,0.060,0.080]}
    
    target_dimensions_dict = {'orange': [0.073,0.074,0.074], 'lemon': [0.074,0.049,0.051], 
                              'lime': [0.048,0.050,0.075], 'tomato': [0.070,0.069,0.069], 
                              'capsicum': [0.094,0.068,0.068], 'potato': [0.065,0.068,0.095], 
                              'pumpkin': [0.082,0.087,0.086], 'garlic': [0.075,0.066,0.064]}
    
    #########

    # estimate target pose using bounding box and robot pose
    target_class = obj_info[0]     # get predicted target label of the box
    target_box = obj_info[1]       # get bounding box measures: [x,y,width,height]
    true_height = target_dimensions_dict[target_class][0]   # look up true height of by class label

    # compute pose of the target based on bounding box info, true object height, and robot's pose
    pixel_height = target_box[3]
    pixel_center = target_box[0]
    distance = true_height/pixel_height * focal_length  # estimated distance between the robot and the centre of the image plane based on height
    # image size 640x480 pixels, 640/2=320
    x_shift = 320/2 - pixel_center              # x distance between bounding box centre and centreline in camera view
    theta = np.arctan(x_shift/focal_length)     # angle of object relative to the robot
    ang = theta + robot_pose[2]     # angle of object in the world frame
    
   # relative object location
    distance_obj = distance/np.cos(theta) # relative distance between robot and object
    x_relative = distance_obj * np.cos(theta) # relative x pose
    y_relative = distance_obj * np.sin(theta) # relative y pose
    relative_pose = {'x': x_relative, 'y': y_relative}
    #print(f'relative_pose: {relative_pose}')

    # location of object in the world frame using rotation matrix
    delta_x_world = x_relative * np.cos(ang) - y_relative * np.sin(ang)
    delta_y_world = x_relative * np.sin(ang) + y_relative * np.cos(ang)
    # add robot pose with delta target pose
    target_pose = {'y': (robot_pose[1]+delta_y_world)[0],
                   'x': (robot_pose[0]+delta_x_world)[0]}
    #print(f'delta_x_world: {delta_x_world}, delta_y_world: {delta_y_world}')
    #print(f'robot_pose_x: {robot_pose[0]}, robot_pose_y: {robot_pose[1]}')
    #print(f'target_pose: {target_pose}')
    return target_pose

def k_means_clustering(dataset, fruit_count, plot = True):
    kmeans = KMeans(n_clusters=fruit_count, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(dataset)
    if plot == True:
        plt.title('K-means clustering (k={})'.format(fruit_count))
        plt.scatter(dataset[:, 0], dataset[:, 1], c=y_kmeans)
        plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s=100, c='red')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    cluster_centre = kmeans.cluster_centers_
    return cluster_centre

def dbscan_clustering(dataset, eps = 0.25, min_samples = 2, plot = True):
    db = DBSCAN(eps = eps, min_samples = min_samples).fit(dataset)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    cluster_centre = np.zeros([n_clusters_,2])
    i = 0
    if plot == True:
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = labels == k

            xy = dataset[class_member_mask & core_samples_mask] 
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=6,  
            )
            if k != -1: 
                cluster_centre[i,:] = np.mean(xy, axis=0)
                plt.plot(
                    cluster_centre[i, 0],
                    cluster_centre[i, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                    
                )
                i += 1
            xy = dataset[class_member_mask & ~core_samples_mask]
            plt.plot(
                xy[:, 0],
                xy[:, 1],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=3,
            )   
        plt.title(f"Estimated number of clusters: {n_clusters_}")
        plt.show()
    return cluster_centre

def merge_estimations(target_pose_dict, method, fruit_count, eps, min_samples, plot = True):
    """
    function:
        merge estimations of the same target
    input:
        target_pose_dict: dict, generated by estimate_pose
    output:
        target_est: dict, target pose estimations after merging
    """
    target_est = {}
    key = []
    x = np.zeros(len(target_pose_dict))
    y = np.zeros(len(target_pose_dict))
    i = 0
    for k, v in target_pose_dict.items():
        x[i] = v['x']
        y[i] = v['y']
        key.append(k)
        i += 1 
    X = np.array(list(zip(x, y))).reshape(len(x), 2)

    fruit_split_dict = {}
    for i in TARGET_TYPES:
        fruit_coord = []
        for j in range(len(X)):
            if key[j][:key[j].index("_")] == i:
                fruit_coord.append(X[j,:])
        fruit_split_dict[i] = fruit_coord
            
    
    for i in TARGET_TYPES:
        dataset = np.array(fruit_split_dict[i])
        if method == 1: #Kmeans clustering
            print("K-means clustering")
            cluster_centre = k_means_clustering(dataset, fruit_count[i], plot)

        elif method == 2: #DBSCAN clustering
            print("DBSCAN clustering")
            cluster_centre = dbscan_clustering(dataset, eps, min_samples, plot)
        
        for j in range(len(cluster_centre)):
            target_est[i + "_" + str(j)] = {'x': cluster_centre[j,0], 'y': cluster_centre[j,1]}
    

    return target_est


# main loop
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))     # get current script directory (TargetPoseEst.py)

    # read in camera matrix
    fileK = f'{script_dir}/calibration/param/intrinsic.txt'
    camera_matrix = np.loadtxt(fileK, delimiter=',')

    # init YOLO model
    model_path = f'{script_dir}/YOLO/model/yolov8_model.pt'
    #Good YOLO models: 4, 6 (really good), 7 
    #Update: YOLO Model 6 is now base model 
    yolo = Detector(model_path)

    # create a dictionary of all the saved images with their corresponding robot pose
    image_poses = {}
    with open(f'{script_dir}/lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']

    # estimate pose of targets in each image
    target_pose_dict = {}
    detected_type_list = []
    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)
        bounding_boxes, bbox_img = yolo.detect_single_image(input_image)
        cv2.imwrite(f'{script_dir}/lab_output/bbox/{image_path.split("/")[-1]}', bbox_img)
        # cv2.imshow('bbox', bbox_img)
        # cv2.waitKey(0)
        robot_pose = image_poses[image_path]

        
        for detection in bounding_boxes:
            # count the occurrence of each target type
            occurrence = detected_type_list.count(detection[0])
            target_pose_dict[f'{detection[0]}_{occurrence}'] = estimate_pose(camera_matrix, detection, robot_pose)

            detected_type_list.append(detection[0])
    
    
    # merge the estimations of the targets so that there are at most 3 estimations of each target type    target_est = {}
    fruit_count = fruit_input()
    plot = input("Plot? (y/n): ")
    if plot == "y" or plot == "Y":
        plot = True 
    else:
        plot = False
    method = input("K-means or DBSCAN? (1/2): ")
    if method == "2":
        eps = float(input("Enter eps: "))
        min_samples = int(input("Enter min_samples: "))
        target_est = merge_estimations(target_pose_dict, int(method), fruit_count, eps, min_samples, plot)
    else:
        target_est = merge_estimations(target_pose_dict, int(method), fruit_count, 0, 0, plot)
    
    print(target_est)
    # save target pose estimations
    with open(f'{script_dir}/lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo, indent=4)

    print('Estimations saved!')

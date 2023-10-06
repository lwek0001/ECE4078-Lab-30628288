import ast

from SLAM_eval import parse_user_map


aruco_markers = str('lab_output/slam.txt')
fruits = open('lab_output/targets.txt', 'r')
fruits = ast.literal_eval(fruits.read())
aruco_markers = parse_user_map(aruco_markers)

EstimateMap = {}
for i in range(1,11):
    EstimateMap[f"aruco{i}_0"] = {"y": float(aruco_markers[i][0]), "x": float(-aruco_markers[i][1])}

EstimateMap.update(fruits)
print(EstimateMap)

with open('EstimateMap.txt', 'w') as f:
    write = f.write(str(EstimateMap))
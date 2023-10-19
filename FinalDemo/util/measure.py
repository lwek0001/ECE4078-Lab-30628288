import numpy as np

class Marker:
    # Measurements are of landmarks in 2D and have a position as well as tag id.
    def __init__(self, position, tag, covariance = (0.1*np.eye(2))):
        self.position = position
        self.tag = tag
        self.covariance = covariance

class Drive:
    # Measurement of the robot wheel velocities
    def __init__(self, left_speed, right_speed, dt, left_cov = 1, right_cov = 1):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov
        if self.left_speed == 0 and self.right_speed == 0: # Stopped
            self.left_cov = 0
            self.right_cov = 0
        elif self.left_speed == self.right_speed:  # Lower covariance since driving straight is consistent
            self.left_cov = 1
            self.right_cov = 1
        else:
            self.left_cov = 2 # Higher covariance since turning is less consistent
            self.right_cov = 2
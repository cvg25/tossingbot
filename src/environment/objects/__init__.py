from src.environment.objects.cube import Cube
from src.environment.objects.ball import Ball
from src.environment.objects.rod import Rod

import pybullet 
import numpy as np

objects = {
    'cube': Cube,
    'ball': Ball,
    'rod': Rod
}

def get_object_in_bounds(obj_name, bounds):
    offset = 0.05
    x = np.random.uniform(bounds[0, 0] + offset, bounds[0, 1] - offset)
    y = np.random.uniform(bounds[1, 0] + offset, bounds[1, 1] - offset)
    z = np.random.uniform(bounds[2, 0] + offset, bounds[2, 1] - offset)
    random_position = [x, y, z]
    random_orientation = pybullet.getQuaternionFromEuler([np.random.uniform() * np.pi,
                                                          np.random.uniform() * np.pi,
                                                          np.random.uniform() * np.pi])

    return objects[obj_name](position=random_position, orientation=random_orientation)

def get_object_at_pose(obj_name, position, orientation):
    if len(orientation) == 3:
        orientation = pybullet.getQuaternionFromEuler(orientation)
    return objects[obj_name](position=position, orientation=orientation)
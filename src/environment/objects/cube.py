from src.environment.utils import get_random_color_rgba
import pybullet
import numpy as np

class Cube:

    def __init__(self, position, orientation, mass=0.01, color=None):
        object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
        object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
        self.id = pybullet.createMultiBody(baseMass=mass, 
                                           baseCollisionShapeIndex=object_shape, 
                                           baseVisualShapeIndex=object_visual, 
                                           basePosition=position, 
                                           baseOrientation=orientation)
        if color is None: color = get_random_color_rgba()
        pybullet.changeVisualShape(self.id, -1, rgbaColor=color)
        
    
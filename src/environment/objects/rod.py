from src.environment.utils import get_random_color_rgba
import pybullet
import numpy as np

class Rod:

    def __init__(self, position, orientation, mass=0.01, color=None):
        radius = 0.015
        length = 0.16
        object_shape = pybullet.createCollisionShape(pybullet.GEOM_CYLINDER, radius=radius, height=length)
        object_visual = pybullet.createVisualShape(pybullet.GEOM_CYLINDER, radius=radius, length=length)
        self.id = pybullet.createMultiBody(baseMass=mass, 
                                           baseCollisionShapeIndex=object_shape, 
                                           baseVisualShapeIndex=object_visual, 
                                           basePosition=position, 
                                           baseOrientation=orientation)
        if color is None: color = get_random_color_rgba()
        pybullet.changeVisualShape(self.id, -1, rgbaColor=color)
        pybullet.changeDynamics(self.id, -1, rollingFriction=0.03)
        
    
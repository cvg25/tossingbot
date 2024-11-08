import pybullet
import numpy as np

class RobotUR5e():

    def __init__(self, base_position, base_orientation):
        self.dt = pybullet.getPhysicsEngineParameters()['fixedTimeStep']
        self.home_joints = [np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        urdf_fpath = './assets/ur5e/ur5e.urdf'
        self.base_position = base_position
        self.base_orientation = base_orientation
        self.id = pybullet.loadURDF(fileName=urdf_fpath, 
                                    basePosition=self.base_position, 
                                    baseOrientation=self.base_orientation,
                                    flags=pybullet.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.joint_infos = [pybullet.getJointInfo(self.id, i) for i in range(pybullet.getNumJoints(self.id))]
        self.joint_ids = [j_info[0] for j_info in self.joint_infos if j_info[2] == pybullet.JOINT_REVOLUTE]
        self.ee_id = 9 # Link ID of UR5 end effector: wrist_3_link-tool0_fixed_joint.
        self.tcp_id = 10  # Link ID of gripper finger tip: tool0_fixed_joint-tool_tip.
        self.reset_joints()

        # Create pedestal base only for visualization purposes.
        pybullet.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=pybullet.createVisualShape(pybullet.GEOM_CYLINDER, radius=0.08, length=self.base_position[2], rgbaColor=[0.5,0.5,0.5,1.0]),
            basePosition=[0., 0., self.base_position[2]/2], 
            baseOrientation=self.base_orientation
        )

    def reset_joints(self):
        for i in range(len(self.joint_ids)):
            pybullet.resetJointState(self.id, self.joint_ids[i], self.home_joints[i])
        self.move_joints(self.home_joints) # Actuate joints to hold that pose.
        for _ in range(240): pybullet.stepSimulation()
        pos, rot = self.get_cartesian_pose()
        self.home_position = pos
        self.home_orientation = pybullet.getEulerFromQuaternion(rot)

    def get_state(self, link_id=None):
        data = pybullet.getLinkState(
            bodyUniqueId=self.id,
            linkIndex=link_id if link_id is not None else self.tcp_id,
            computeForwardKinematics=True,
            computeLinkVelocity=True)
        position, orientation = data[:2]
        linear_velocity = data[-2]
        return position, orientation, linear_velocity
    
    def get_cartesian_pose(self):
        position, orientation, _ = self.get_state()
        return position, orientation
    
    def get_joints(self):
        position_velocities = np.array([pybullet.getJointState(self.id, j_id)[:2] for j_id in self.joint_ids])
        positions = position_velocities[:,0]
        velocities = position_velocities[:,1]
        return positions, velocities

    def move_joints(self, j_targets):
        """Move to target joint positions with position control."""
        pybullet.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=j_targets,
            positionGains=[0.01]*6)

    def compute_inverse_kinematics(self, target_pos, target_rot):
        if len(target_rot) == 3: # convert to quat
            target_rot = pybullet.getQuaternionFromEuler(target_rot)
        joints = pybullet.calculateInverseKinematics(
            bodyUniqueId=self.id,
            endEffectorLinkIndex=self.tcp_id,
            targetPosition=target_pos,
            targetOrientation=target_rot,
            maxNumIterations=100)[:len(self.joint_ids)]
        return joints

    def move_pose(self, position, orientation):
        """Move to target end effector position."""
        joints = self.compute_inverse_kinematics(target_pos=position, target_rot=orientation)
        self.move_joints(j_targets=joints)

    def move_joints_velocity(self, j_vel_targets):
        """Move to target joint positions with velocity control."""
        pybullet.setJointMotorControlArray(
            bodyIndex=self.id,
            jointIndices=self.joint_ids,
            controlMode=pybullet.VELOCITY_CONTROL,
            targetVelocities=j_vel_targets)
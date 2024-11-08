from src.environment.robot_ur5e import RobotUR5e
from src.environment.robotiq2f85 import Robotiq2F85
from src.environment.camera_rgbd import CameraRGBD
from src.environment.utils import get_random_color_rgba, seed_everything
from src.environment.objects import get_object_in_bounds, get_object_at_pose

import pybullet
import numpy as np
import os
import time 

class Environment():

    def __init__(self, seed=47, mode=pybullet.GUI):
        seed_everything(seed=seed)
        self.dt = 1/480.0
        self.sim_step = 0
        self.wait = True

        # Configure and start pybullet
        self.id = pybullet.connect(mode) # mode is pybullet.DIRECT or pybullet.GUI (default) for local GUI.
        pybullet.setPhysicsEngineParameter(enableFileCaching=0)
        pybullet.setAdditionalSearchPath(str(os.getcwd()))
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.setTimeStep(self.dt)

        # Entities
        self.robot_base_pos = [0.,0.,0.3] # X, Y, Z
        self.robot_base_rot = pybullet.getQuaternionFromEuler([0.,0.,0.]) # RX, RY, RZ
        self.robot = None
        self.gripper = None

        # Objects
        self.workspace_plane_dims = [0.6, 0.6, 0.2]
        self.workspace_plane_center_pos = [0.5, 0.0, 0.0]
        x_min = self.workspace_plane_center_pos[0] - self.workspace_plane_dims[0]/2
        x_max = self.workspace_plane_center_pos[0] + self.workspace_plane_dims[0]/2
        y_min = self.workspace_plane_center_pos[1] - self.workspace_plane_dims[1]/2
        y_max = self.workspace_plane_center_pos[1] + self.workspace_plane_dims[1]/2
        z_min = self.workspace_plane_center_pos[-1]
        z_max = self.workspace_plane_dims[-1]
        # 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining region in 3D space of the grasp workspace.
        self.workspace_bounds = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
        self.objects = []
        self.patience = 12

        # Data From Paper: https://arxiv.org/pdf/1903.11239
        self.c_h = 0.04
        self.c_d = 0.7

        # Goals
        self.box_side_y = 0.15
        self.box_side_x = 0.25
        x_start = 1.5
        x_nboxes = 3
        y_start = -0.225
        y_nboxes = 4
        self.x_goals = np.linspace(x_start, x_start+self.box_side_x*(x_nboxes-1), x_nboxes)
        self.y_goals = np.linspace(y_start, y_start+self.box_side_y*(y_nboxes-1), y_nboxes)
        self.goal_viz_ids = []
        # Camera
        self.camera = CameraRGBD(position=[0.5, 0, 0.55], orientation=[0, np.pi, np.pi/2], noise=True)
        self.PIXEL_SIZE = 0.005       
        self.heightmap = None
        self.colormap = None
        self.xyzmap = None

        self.reset()

    def disable_rendering(self):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

    def enable_rendering(self):
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

    def toggle_wait(self):
        self.wait = not self.wait

    def reset(self):
        pybullet.resetSimulation(pybullet.RESET_USE_DEFORMABLE_WORLD)
        pybullet.setGravity(0, 0, -9.8)
        # Temporarily disable rendering to load URDFs faster.
        self.disable_rendering()

        # Load plane.
        pybullet.loadURDF('assets/plane/plane.urdf', [0, 0, -0.001])

        # Load robot.
        self.robot = RobotUR5e(base_position=self.robot_base_pos, base_orientation=self.robot_base_rot)

        # Load gripper.
        if self.gripper is not None:
            self.gripper.running = False
            while self.gripper.constraints_thread.is_alive():
                time.sleep(0.001)
        self.gripper = Robotiq2F85(robot_id=self.robot.id, ee_link_id=self.robot.ee_id)
        self.gripper.release()

        # Load workspace.
        offset = 0.08
        dims = [self.workspace_plane_dims[0]/2 + offset, self.workspace_plane_dims[0]/2 + offset, 0.001]
        plane_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=dims)
        plane_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=dims)
        plane_id = pybullet.createMultiBody(0, plane_shape, plane_visual, basePosition=self.workspace_plane_center_pos)
        pybullet.changeVisualShape(plane_id, -1, rgbaColor=[0.2,0.2,0.2,1.0])
        # Set workspace walls
        bound_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[self.workspace_plane_dims[0]/2 + offset, 0.005, 0.04])
        bound_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[self.workspace_plane_dims[0]/2 + offset, 0.005, 0.04])
        b_id = pybullet.createMultiBody(0, bound_shape, bound_visual, basePosition=[self.workspace_plane_center_pos[0], self.workspace_bounds[1,0] - offset, 0.02])
        pybullet.changeVisualShape(b_id, -1, rgbaColor=[0.2,0.2,0.2,1.0])

        bound_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[self.workspace_plane_dims[0]/2 + offset, 0.005, 0.04])
        bound_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[self.workspace_plane_dims[0]/2 + offset, 0.005, 0.04])
        b_id = pybullet.createMultiBody(0, bound_shape, bound_visual, basePosition=[self.workspace_plane_center_pos[0], self.workspace_bounds[1,1] + offset, 0.02])
        pybullet.changeVisualShape(b_id, -1, rgbaColor=[0.2,0.2,0.2,1.0])

        bound_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.005, self.workspace_plane_dims[0]/2 + offset, 0.04])
        bound_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.005, self.workspace_plane_dims[0]/2 + offset, 0.04])
        b_id = pybullet.createMultiBody(0, bound_shape, bound_visual, basePosition=[self.workspace_bounds[0,0] - offset, self.workspace_plane_center_pos[1], 0.02])
        pybullet.changeVisualShape(b_id, -1, rgbaColor=[0.2,0.2,0.2,1.0])

        bound_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.005, self.workspace_plane_dims[0]/2 + offset, 0.04])
        bound_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.005, self.workspace_plane_dims[0]/2 + offset, 0.04])
        b_id = pybullet.createMultiBody(0, bound_shape, bound_visual, basePosition=[self.workspace_bounds[0,1] + offset, self.workspace_plane_center_pos[1], 0.02])
        pybullet.changeVisualShape(b_id, -1, rgbaColor=[0.2,0.2,0.2,1.0])

        # Load goal boxes.
        self._load_goals()
        self.goal = self._sample_goal()
        self._draw_goal()

        # Add objects
        self._remove_all_objects()
        self._populate_objects()

        # Re-enable rendering.
        self.enable_rendering()
        self._step_simulation(n_steps=600)
        return self._get_observation(), self.goal 
    
    def _step_simulation(self, n_steps):
        for _ in range(n_steps):
            pybullet.stepSimulation() 
            if self.wait: time.sleep(self.dt)

    def _populate_objects(self):
        for _ in range(4):
            self.objects.append(get_object_in_bounds(obj_name='ball', bounds=self.workspace_bounds))
            self.objects.append(get_object_in_bounds(obj_name='cube', bounds=self.workspace_bounds))
            self.objects.append(get_object_in_bounds(obj_name='rod', bounds=self.workspace_bounds))
        #self.objects.append(get_object_at_pose(obj_name='cube', position=[1.2, 0.2, 0.7], orientation=[0,0,0]))
        self.no_change_count = 0

    def _load_goals(self):
        xs, ys = np.meshgrid(self.x_goals, self.y_goals)
        for x, y in zip(xs.ravel(), ys.ravel()):
            self._load_obj(file_path='assets/target_box/box.obj', position=[x, y, 0], orientation=[0,0,np.pi/2], rgba=[0.5, 0.37, 0.36, 1.0], scale=0.01)

    def _load_obj(self, file_path, position, orientation, rgba, scale, mass=0):
        id_visual = pybullet.createVisualShape(shapeType=pybullet.GEOM_MESH, 
                                               fileName=file_path,
                                               rgbaColor=rgba,
                                               meshScale=[scale]*3)
        id_collision = pybullet.createCollisionShape(shapeType=pybullet.GEOM_MESH,
                                                     fileName=file_path,
                                                     meshScale=[scale]*3,
                                                     flags=pybullet.GEOM_FORCE_CONCAVE_TRIMESH)
        pybullet.createMultiBody(baseMass=mass,
                                 baseCollisionShapeIndex=id_collision,
                                 baseVisualShapeIndex=id_visual,
                                 basePosition=position,
                                 baseOrientation=orientation if len(orientation) == 4 else pybullet.getQuaternionFromEuler(orientation))

    def _get_observation(self):
        color, depth, position, orientation, intrinsics = self.camera.get_state()

        pointcloud = CameraRGBD.get_pointcloud(depth, intrinsics)
        position = np.float32(position).reshape(3,1)
        rotation = pybullet.getMatrixFromQuaternion(orientation)
        rotation = np.float32(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        pointcloud = CameraRGBD.transform_pointcloud(pointcloud, transform)
        heightmap, colormap, xyzmap = CameraRGBD.get_heightmap(pointcloud, color, self.workspace_bounds, self.PIXEL_SIZE)
        self.heightmap = heightmap
        self.colormap = colormap
        self.xyzmap = xyzmap
        rgbd_heightmap = np.concatenate((colormap, heightmap.reshape(*heightmap.shape, 1)), axis=-1)
        return rgbd_heightmap

    def _sample_goal(self):
        x = np.random.choice(self.x_goals)
        y = np.random.choice(self.y_goals)
        z = 0.0
        return [x,y,z]
    
    def _draw_goal(self):
        # Remove old goal.
        for id in self.goal_viz_ids:
            pybullet.removeBody(id)
        self.goal_viz_ids = []

        # Draw new goal.
        xs = [self.goal[0], self.goal[0] + self.box_side_x / 2, self.goal[0], self.goal[0] - self.box_side_x/ 2]
        ys = [self.goal[1] + self.box_side_y / 2, self.goal[1], self.goal[1] - self.box_side_y / 2, self.goal[1]]
        rotzs = [0,np.pi/2,0,np.pi/2]
        visual_side_y = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[self.box_side_y/2+0.002, 0.004, 0.001], rgbaColor=[0.0,1.0,0.0,1.0])    
        visual_side_x = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[self.box_side_x/2+0.002, 0.004, 0.001], rgbaColor=[0.0,1.0,0.0,1.0])    
        for data in zip(xs, ys, rotzs):
            x, y, rotz = data
            position = [x, y, self.goal[2]+0.21]
            orientation = pybullet.getQuaternionFromEuler([0., 0., rotz])
            visual_side = visual_side_y if rotz == np.pi/2 else visual_side_x
            side_id = pybullet.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual_side, basePosition=position, baseOrientation=orientation)
            self.goal_viz_ids.append(side_id)
    
    def decode_action(self, action):
        grasp_x, grasp_y, grasp_z = self.xyzmap[action['coord_height'], action['coord_width']]
        grasp_zrot = np.deg2rad(action['z_rotation'])
        grasp_pose = (grasp_x, grasp_y, grasp_z, grasp_zrot)
        throw_velocity = action['velocity']
        return grasp_pose, throw_velocity

    def step(self, action):
        # -1. Save current state of the environment before taking action.
        true_landing_pos = None
        throw_physics_failure = False
        obj_positions_start = self._get_object_positions()
        
        # 0. Decode action
        grasp_pose, throw_velocity = self.decode_action(action)

        # 1. Execute grasping primitive.
        self.grasping_primitive(x_pos=grasp_pose[0],
                                y_pos=grasp_pose[1],
                                z_pos=grasp_pose[2],
                                z_rot=grasp_pose[3])
        obj_grasped_id = self._get_grasped_object()
        grasp_success = obj_grasped_id is not None
        # 2. If grasp success, then execute throwing primitive.
        if grasp_success:
            self.throwing_primitive(v_release=throw_velocity)
            true_landing_pos, throw_physics_failure = self.track_object_landing_pos(obj_id=obj_grasped_id)
            self._step_simulation(n_steps=400)
            throw_success = self._check_throw_success(obj_id=obj_grasped_id)
            """
            As we show in Sec. VI-E, supervising grasps by the accuracy
            of throws eventually leads to more stable grasps and better
            overall throwing performance. The grasping policy learns to
            favor grasps that lead to successful throws, which is a stronger
            requirement than simple grasp success.
            """
        else:
            throw_success = False
            self.gripper.release()
        # 3. Go back to home pose.
        self.robot.move_joints(self.robot.home_joints)
        self._step_simulation(n_steps=600) 

        # 4. Setup environment for the next trial.    
        reward = [grasp_success, throw_success]
        info = {
            'true_landing_pos': true_landing_pos,
            'throw_physics_failure': throw_physics_failure
        }

        self.disable_rendering()
        # - Sample new goal.
        self.goal = self._sample_goal()
        self._draw_goal()
        # - Detect object position changes.
        obj_positions_current = self._get_object_positions()
        obj_positions_dist = np.linalg.norm(obj_positions_current-obj_positions_start, 1)
        obj_positions_changed = (obj_positions_dist > 0.02).any()
        self.no_change_count = 0 if obj_positions_changed else self.no_change_count + 1
        if self.no_change_count > self.patience: # if no changes occurred after 10 trials, reset environment.
            print(f'No change detected for more than {self.patience} trials... resetting environment.')
            self._remove_all_objects()
            self.no_change_count = 0
        else:
            self._remove_objects_outside_workspace()
        # - Repopulate workspace with objects when empty.
        if len(self.objects) == 0:
            self._populate_objects()
        self.enable_rendering()    
        self._step_simulation(n_steps=600) 
        
        return self._get_observation(), self.goal, reward, info

    def grasping_primitive(self, x_pos, y_pos, z_pos, z_rot):
        """Executes a parameterized grasping primitive."""
        xyz_pos = [x_pos, y_pos, z_pos]
        xyz_rot = np.array(self.robot.home_orientation)
        xyz_rot[-1] = xyz_rot[-1] + z_rot
        # Open the gripper
        self.gripper.release()

        # Move robot down on Z.
        pos = np.array(self.robot.home_position)
        pos[-1] = 0.3
        self.robot.move_pose(position=pos, orientation=xyz_rot)
        self._step_simulation(n_steps=400)

        # Go to pre-grasp pose
        xyz_pos_pre = np.array(xyz_pos)
        xyz_pos_pre[-1] = xyz_pos_pre[-1] + 0.15
        self.robot.move_pose(position=xyz_pos_pre, orientation=xyz_rot)
        self._step_simulation(n_steps=600)

        # Go to grasp pose
        xyz_pos_grasp = np.array(xyz_pos)
        xyz_pos_grasp[-1] = max(xyz_pos_grasp[-1] - 0.04, 0.003) # prevent hitting the floor.
        self.robot.move_pose(position=xyz_pos_grasp, orientation=xyz_rot)
        self._step_simulation(n_steps=600)

        # Close gripper to pick object
        self.gripper.close()
        self._step_simulation(n_steps=700)

        # Lift object
        lift_pos = [xyz_pos_grasp[0], xyz_pos_grasp[1], 0.20]
        self.robot.move_pose(position=lift_pos, orientation=xyz_rot)
        self._step_simulation(n_steps=600)

        pos = np.array(self.robot.home_position)
        pos[-1] = 0.3
        xyz_rot = np.array(self.robot.home_orientation)
        self.robot.move_pose(position=pos, orientation=xyz_rot)
        self._step_simulation(n_steps=600)

    def throwing_primitive(self, v_release):
        """Executes a parameterized throwing primitive."""
        px = self.goal[0]
        py = self.goal[1]
        # Compute throwing release position
        angle_correction_rad = self._compute_throwing_angle_correction(x_goal=px, y_goal=py)
        angle = np.arctan(py/px) - angle_correction_rad
        rx = self.c_d * np.cos(angle)
        ry = self.c_d * np.sin(angle)
        rz = self.c_h + self.robot.base_position[2]
        assert np.abs(np.sqrt(rx**2+ry**2) - self.c_d) < 0.01, f"Error, np.sqrt(rx**2+ry**2) not equals self.c_d. {np.abs(np.sqrt(rx**2+ry**2))}"
        # Compute throwing orientation
        rotx = 0
        roty = np.deg2rad(135) # 90+45 Constrain the direction of v to be angled 45 deg upwards.
        rotz = 0

        # Throwing joints
        joints_throw = self.robot.compute_inverse_kinematics(target_pos=[rx, ry, rz], target_rot=[rotx, roty, rotz])
        joints_throw = np.array(joints_throw)
        joints_throw[0] = np.pi + angle
        joints_throw[4] = -np.pi/2
        joints_throw[5] = 0
        # Set prethrow pose.
        joints_prethrow = np.array(joints_throw)
        joints_prethrow[2] = joints_prethrow[2] + np.deg2rad(30)
        # Set post-throw pose.
        joints_postthrow = np.array(joints_throw)
        joints_postthrow[2] = joints_postthrow[2] - np.deg2rad(30)
        # Compute throwing velocity.
        r = 0.61835 # movement radious from origin to tcp.
        w = v_release / r # Angular release velocity (since is a joint move) from v = w*r
        
        # Throwing primitive execution.
        # 1. Go to pre-throw joints.
        self.robot.move_joints(j_targets=joints_prethrow)
        self._step_simulation(n_steps=400)
        # 2. Start throwing movement with target velocity.
        self.robot.move_joints_velocity(j_vel_targets=[0,0,-w,0,0,0])
        # 3. Check joint position to open gripper.
        while (pybullet.getJointState(self.robot.id, self.robot.joint_ids[2])[0]-joints_throw[2]) > np.deg2rad(0.001):
            self._step_simulation(n_steps=1)
        self.gripper.release()        
        # 4. Terminate the throwing primitive at the post-throw joints.
        while (pybullet.getJointState(self.robot.id, self.robot.joint_ids[2])[0]-joints_postthrow[2]) > np.deg2rad(0.001):
            self._step_simulation(n_steps=1)
        self.robot.move_joints(joints_postthrow)

    def track_object_landing_pos(self, obj_id, max_steps=400):
        throw_physics_failure = False
        obj_pos = np.array(pybullet.getBasePositionAndOrientation(obj_id)[0])
        track_steps = 0

        while (not throw_physics_failure and obj_pos[-1] > 0.05) and track_steps < max_steps:
            self._step_simulation(n_steps=1)
            obj_pos = np.array(pybullet.getBasePositionAndOrientation(obj_id)[0])
            throw_physics_failure = obj_pos[-1] > 0.85
            track_steps += 1
        true_landing_pos = np.array(pybullet.getBasePositionAndOrientation(obj_id)[0])
        if not throw_physics_failure:
            throw_physics_failure = true_landing_pos[0] < 1.25
        return true_landing_pos, throw_physics_failure

    def _check_throw_success(self, obj_id):
        obj_pos = np.array(pybullet.getBasePositionAndOrientation(obj_id)[0])
        x, y, _ = self.goal
        x_min = x - self.box_side_x/2
        x_max = x + self.box_side_x/2
        y_min = y - self.box_side_y/2
        y_max = y + self.box_side_y/2
        x_obj = obj_pos[0]
        y_obj = obj_pos[1]
        return x_obj > x_min and x_obj < x_max and y_obj > y_min and y_obj < y_max

    def _remove_objects_outside_workspace(self):
        x_min = self.workspace_bounds[0][0]
        x_max = self.workspace_bounds[0][1]
        y_min = self.workspace_bounds[1][0]
        y_max = self.workspace_bounds[1][1]
        self.objects_in_workspace = []
        while self.objects:
            obj = self.objects.pop()
            obj_pos = np.array(pybullet.getBasePositionAndOrientation(obj.id)[0])
            if not (obj_pos[0] > x_min and obj_pos[0] < x_max and obj_pos[1] > y_min and obj_pos[1] < y_max):
                pybullet.removeBody(obj.id)
            else:
                self.objects_in_workspace.append(obj)
        self.objects = self.objects_in_workspace

    def check_sim(self):
        x_min = self.workspace_bounds[0][0]
        x_max = self.workspace_bounds[0][1]
        y_min = self.workspace_bounds[1][0]
        y_max = self.workspace_bounds[1][1]
        gripper_pos = np.array(pybullet.getBasePositionAndOrientation(self.gripper.body)[0])
        reset_sim = False
        if not (gripper_pos[0] > x_min and gripper_pos[0] < x_max and gripper_pos[1] > y_min and gripper_pos[1] < y_max):
            reset_sim = True
        return reset_sim

    def _remove_all_objects(self):
        for obj in self.objects:
            pybullet.removeBody(obj.id)
        self.objects = []

    def _get_object_positions(self):
        obj_positions = []
        for obj in self.objects:
            obj_pos = np.array(pybullet.getBasePositionAndOrientation(obj.id)[0])
            obj_positions.append(obj_pos)
        return np.array(obj_positions)

    def _get_grasped_object(self):
        obj_id = None
        if self.gripper.grasp_width() > 0.001:
            ee_pos = np.array(self.robot.get_cartesian_pose()[0])
            for obj in self.objects:
                obj_pos = np.array(pybullet.getBasePositionAndOrientation(obj.id)[0])
                if np.linalg.norm(ee_pos-obj_pos) < 0.095:
                    obj_id = obj.id
                    break
        return obj_id

    def _compute_throwing_angle_correction(self, x_goal, y_goal):
        r = 0.138 # distance between base link center and shoulder joint.
        Ox, Oy = x_goal, y_goal # goal coordinate
        Bx, By = 0, 0 # Base origin.
        Tx, Ty = 0, r # Throwing origin.
        OB = np.array([Bx-Ox, By-Oy]).reshape((1,2))
        OT = np.array([Tx-Ox, Ty-Oy]).reshape((1,2))
        angle_rad = np.arccos((OB @ OT.T) / (np.linalg.norm(OB) * np.linalg.norm(OT)))
        return angle_rad
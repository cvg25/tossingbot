import pybullet
import numpy as np

class CameraRGBD():
    #https://colab.research.google.com/drive/1jAyhumd7DTxJB2oZufob9crVxETAEKbV?usp=sharing#scrollTo=0bj91lBq6WeK
    def __init__(self, position, orientation, noise):
        # Camera parameters.
        self.position = position
        self.orientation = pybullet.getQuaternionFromEuler(orientation)
        self.noise = noise
        self.img_size = (480, 848)
        self.intrinsics = (424.790, 0., 424.947, 0., 424.790, 237.767, 0, 0, 1)  

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pybullet.getMatrixFromQuaternion(self.orientation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        focal_len = self.intrinsics[0]
        self.znear, self.zfar = (0.01, 10.)
        self.viewm = pybullet.computeViewMatrix(position, lookat, updir)
        fovh = (self.img_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = self.img_size[1] / self.img_size[0]
        self.projm = pybullet.computeProjectionMatrixFOV(fovh, aspect_ratio, self.znear, self.zfar)

    def get_state(self):
        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pybullet.getCameraImage(
            width=self.img_size[1],
            height=self.img_size[0],
            viewMatrix=self.viewm,
            projectionMatrix=self.projm,
            shadow=0,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_TINY_RENDERER)

        # Get color image.
        color_image_size = (self.img_size[0], self.img_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if self.noise:
            color = np.int32(color)
            color += np.int32(np.random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))
        color = color / 255
        
        # Get depth image.
        depth_image_size = (self.img_size[0], self.img_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (self.zfar + self.znear - (2 * zbuffer - 1) * (self.zfar - self.znear))
        depth = (2 * self.znear * self.zfar) / depth
        if self.noise:
            depth += np.random.normal(0, 0.003, depth.shape)

        intrinsics = np.float32(self.intrinsics).reshape(3, 3)
        return color, depth, self.position, self.orientation, intrinsics
    
    @staticmethod
    def get_pointcloud(depth, intrinsics):
        """Get 3D pointcloud from perspective depth image.
        Args:
          depth: HxW float array of perspective depth in meters.
          intrinsics: 3x3 float array of camera intrinsics matrix.
        Returns:
          points: HxWx3 float array of 3D points in camera coordinates.
        """
        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
        points = np.float32([px, py, depth]).transpose(1, 2, 0)
        return points
    
    @staticmethod
    def transform_pointcloud(points, transform):
        """Apply rigid transformation to 3D pointcloud.
        Args:
          points: HxWx3 float array of 3D points in camera coordinates.
          transform: 4x4 float array representing a rigid transformation matrix.
        Returns:
          points: HxWx3 float array of transformed 3D points.
        """
        padding = ((0, 0), (0, 0), (0, 1))
        homogen_points = np.pad(points.copy(), padding, 'constant', constant_values=1)
        for i in range(3):
          points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
        return points
    
    @staticmethod
    def get_heightmap(points, colors, bounds, pixel_size):
        """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.
        Args:
          points: HxWx3 float array of 3D points in world coordinates.
          colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
          bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
            region in 3D space to generate heightmap in world coordinates.
          pixel_size: float defining size of each pixel in meters.
        Returns:
          heightmap: HxW float array of height (from lower z-bound) in meters.
          colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
          xyzmap: HxWx3 float array of XYZ points in world coordinates.
        """
        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
        heightmap = np.zeros((height, width), dtype=np.float32)
        colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.float32)
        xyzmap = np.zeros((height, width, 3), dtype=np.float32)

        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
        iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
        iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = colors[valid]
        
        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]
        px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        heightmap[py, px] = points[:, 2] - bounds[2, 0]
        for c in range(colors.shape[-1]):
          colormap[py, px, c] = colors[:, c]
          xyzmap[py, px, c] = points[:, c]
        colormap = colormap[::-1, :, :]  # Flip up-down.
        xv, yv = np.meshgrid(np.linspace(bounds[0, 0], bounds[0, 1], width),
                             np.linspace(bounds[1, 0], bounds[1, 1], height))
        xyzmap[:, :, 0] = xv
        xyzmap[:, :, 1] = yv
        xyzmap = xyzmap[::-1, :, :]  # Flip up-down.
        heightmap = heightmap[::-1, :]  # Flip up-down.
        return heightmap, colormap, xyzmap
    
    @staticmethod
    def xyz_to_pix(position, bounds, pixel_size):
        """Convert from 3D position to pixel location on heightmap."""
        u = int(np.round((bounds[1, 1] - position[1]) / pixel_size))
        v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
        return (u, v)
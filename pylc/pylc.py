import os
import sys
import math
import numpy as np

from transformations import euler_matrix

build_dir = os.path.dirname(os.path.abspath(__file__)) + '/build'
sys.path.append(build_dir)  # should contain pylc_lib compiled .so file
import pylc_lib

def compute_camera_instrics_matrix(fov_deg, w, h):
    """
    Args:
        fov_deg: (float) FOV angle in degrees.
        w: (int) width in pixels.
        h: (int) height in pixels.
    Returns: (np.ndarray, np.float32, shape=(3, 3))
             camara intrinsics matrix.
    """
    fov_rad = fov_deg * math.pi / 180.0
    fx = fy = (w / 2) / math.tan(fov_rad / 2)
    cx, cy = w / 2, h / 2

    return np.array([[fx, 0., cx],
                     [0., fy, cy],
                     [0., 0., 1.]], dtype=np.float32)

class LCDevice:
    def __init__(self, LASER_PARAMS=None, CAMERA_PARAMS=None):
        
        # These are the defaults.
        # The extrinsics are set such that
        #     - Velo's coordinate is treated as world coordinate.
        #     - The camera has the same coordinate frame as the lidar, including
        #       same origin, only the axes have been permuted.
        #         - camera's X lies along velo's -Y
        #         - camera's Y lies along velo's -X
        #         - camera's Z lies along velo's +X.
        #       - This is given by roll, pitch, yaw of -90, 0, -90.
        #     - laser lies 20cm to the right of camera (0.2 along velo's -Y)
        self.LASER_PARAMS = dict(
            x=0.0, y=-0.2, z=0.0, # 20cm to right of camera
            roll=-90.0, pitch=0.0, yaw=-90.0,
            fov=80,
            galvo_m=-2.2450289e+01,
            galvo_b=-6.8641598e-01,
            maxADC=15000,
            thickness=0.00055,
            divergence=0.11/2.,
            laser_limit=14000,
            laser_timestep=1.5e-5
        )
        # Merge with defaults.
        if LASER_PARAMS is not None:
            for key in self.LASER_PARAMS:
                if key in LASER_PARAMS:
                    self.LASER_PARAMS[key] = LASER_PARAMS[key]

        self.CAMERA_PARAMS = dict(
            x=0.0, y=0.0, z=0.0,
            roll=-90.0, pitch=0.0, yaw=-90.0,
            width=512, height=512,
            limit=1.0,
            distortion=[-0.033918, 0.027494, -0.001691, -0.001078, 0.000000],
            fov=80,
            matrix=None
        )
        # Merge with defaults.
        if CAMERA_PARAMS is not None:
            for key in self.CAMERA_PARAMS:
                if key in CAMERA_PARAMS:
                    self.CAMERA_PARAMS[key] = CAMERA_PARAMS[key]

        # Compute camera matrix if not already specified.
        if self.CAMERA_PARAMS['matrix'] is None:
            self.CAMERA_PARAMS['matrix'] = compute_camera_instrics_matrix(
                self.CAMERA_PARAMS['fov'],
                self.CAMERA_PARAMS['width'],
                self.CAMERA_PARAMS['height']
            )

        # Compute transforms.
        self.TRANSFORMS = self._compute_transforms()

        # Creating laser datum.
        l_datum = pylc_lib.Datum()
        l_datum.type = 'laser'
        l_datum.laser_name = u'laser01'
        l_datum.fov = self.LASER_PARAMS['fov']
        l_datum.galvo_m = self.LASER_PARAMS['galvo_m']
        l_datum.galvo_b = self.LASER_PARAMS['galvo_b']
        l_datum.maxADC = self.LASER_PARAMS['maxADC']
        l_datum.thickness = self.LASER_PARAMS['thickness']
        l_datum.divergence = self.LASER_PARAMS['divergence']
        l_datum.laser_limit = self.LASER_PARAMS['laser_limit']
        l_datum.laser_timestep = self.LASER_PARAMS['laser_timestep']

        # Creating camera datum.
        c_datum = pylc_lib.Datum()
        c_datum.type = 'camera'
        c_datum.camera_name = u'camera01'
        c_datum.rgb_matrix = self.CAMERA_PARAMS['matrix']
        c_datum.limit = self.CAMERA_PARAMS['limit']
        c_datum.depth_matrix = self.CAMERA_PARAMS['matrix']
        c_datum.cam_to_world = self.TRANSFORMS['cTw']
        
        c_datum.cam_to_laser = {u'laser01': self.TRANSFORMS['lTc']}  # TODO: figure out why cam_to_laser is assigned lTc
        c_datum.fov = self.CAMERA_PARAMS['fov']
        c_datum.distortion = np.array(self.CAMERA_PARAMS['distortion'], dtype=np.float32).reshape(1,5)
        c_datum.imgh = self.CAMERA_PARAMS['height']
        c_datum.imgw = self.CAMERA_PARAMS['width']

        self.datum_processor = pylc_lib.DatumProcessor()
        self.datum_processor.setSensors([c_datum], [l_datum])
    
    def _get_transform_from_xyzrpy(self, x, y, z, roll, pitch, yaw):
        # convert roll, pitch, yaw to radians.
        roll, pitch, yaw = roll * np.pi/180., pitch * np.pi/180., yaw * np.pi/180.
        transform = euler_matrix(roll, pitch, yaw)
        transform[:3, 3] = [x, y, z]
        return transform.astype(np.float32)

    def _compute_transforms(self):
        TRANSFORMS = {}
    
        TRANSFORMS['cTw'] = self._get_transform_from_xyzrpy(self.CAMERA_PARAMS['x'],
                                                            self.CAMERA_PARAMS['y'],
                                                            self.CAMERA_PARAMS['z'],
                                                            self.CAMERA_PARAMS['roll'],
                                                            self.CAMERA_PARAMS['pitch'],
                                                            self.CAMERA_PARAMS['yaw'])
        TRANSFORMS['wTc'] = np.linalg.inv(TRANSFORMS['cTw'])
        
        TRANSFORMS['lTw'] = self._get_transform_from_xyzrpy(self.LASER_PARAMS['x'],
                                                            self.LASER_PARAMS['y'],
                                                            self.LASER_PARAMS['z'],
                                                            self.LASER_PARAMS['roll'],
                                                            self.LASER_PARAMS['pitch'],
                                                            self.LASER_PARAMS['yaw'])
        TRANSFORMS['wTl'] = np.linalg.inv(TRANSFORMS['lTw'])
        
        TRANSFORMS['lTc'] = TRANSFORMS['wTc'].dot(TRANSFORMS['lTw'])
        TRANSFORMS['cTl'] = np.linalg.inv(TRANSFORMS['lTc'])
        
        return TRANSFORMS

    def get_return(self, np_depth_image, np_design_pts):
        """
        Args:
            np_depth_image: (np.ndarray, dtype=float32, shape=(H, W))
            np_design_pts: (np.ndarray, dtype=float32, shape=(N, 2))
                           Axis 1 corresponds to x, z in LC camera frame.
        Returns:
            lc_output_image: (np.ndarray, dtype=float32, shape=(H, W, 4))) LC image.
                             - Channels denote (x, y, z, intensity).
                             - Pixels that aren't a part of LC return will have NaNs in one of
                               the 4 channels.
                             - Intensity ranges from 0. to 255.
        """
        pylc_input = pylc_lib.Input()
        pylc_input.camera_name = u'camera01'
        pylc_input.depth_image = np_depth_image

        dx = np_design_pts[:, [0]]
        dy = np.zeros((len(np_design_pts), 1), dtype=np.float32)
        dz = np_design_pts[:, [1]]
        ones = np.ones((len(np_design_pts), 1), dtype=np.float32)
        np_design_pts = np.hstack((dx, dy, dz, ones))
        pylc_input.design_pts_multi = [np_design_pts]  # TODO: change this to pylc_input.design_points

        pylc_output = pylc_lib.Output()
        pylc_lib.processPointsJoint(self.datum_processor,
                                [pylc_input],
                                {u'camera01': 0},
                                [{u'C': u'camera01', u'L': u'laser01'}],
                                pylc_output,
                                False)
        
        # LC output.
        assert len(pylc_output.images_multi) == 1
        assert len(pylc_output.images_multi[0]) == 1
        # np.ndarray, dtype=np.float32, shape=(512, 512, 4)
        output_image = pylc_output.images_multi[0][0]

        return output_image

if __name__ == '__main__':
    import numpy as  np
    import matplotlib; matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    LASER_PARAMS = dict(
        x=0.0,
        y=-2.0,
        z=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        fov=40
    )

    CAMERA_PARAMS = dict(
        x=-0.2,
        y=-2.0,
        z=0.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        width=512,
        height=512,
        limit=1.0,
        distortion=[-0.033918, 0.027494, -0.001691, -0.001078, 0.000000],
        fov=80,
    )

    # Create LCDevice.
    lc_device = LCDevice(LASER_PARAMS, CAMERA_PARAMS)

    depth_image = np.load('data/depth_image.npy')
    rgb_image = np.load('data/rgb_image.npy')
    design_points = np.load('data/design_points.npy')
    design_points = design_points[:, [0, 2]]

    plt.imshow(depth_image); plt.show()
    plt.imshow(rgb_image); plt.show()

    output_image = lc_device.get_return(depth_image, design_points)

    # Show 4-channel image.
    plt.imshow(output_image); plt.title('4-channel image.'); plt.show()

    # Show intensity image.
    intensity = output_image[:, :, 3]
    intensity[np.isnan(intensity)] = 255
    plt.imshow(intensity); plt.title('Intensities. NaNs are yellow.'); plt.show()

    # Output cloud: reshape image and remove points with NaNs.
    output_cloud = output_image.reshape(-1, 4)
    not_nan_mask = np.all(np.isfinite(output_cloud), axis=1)
    output_cloud = output_cloud[not_nan_mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = output_cloud[np.random.choice(len(output_cloud), 5000)]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3])
    ax.set_xlabel('X Label'); ax.set_ylabel('Y Label'); ax.set_zlabel('Z Label')
    ax.set_ylim(-3.5, 3.5)
    ax.set_zlim(0, 5.5)
    plt.title('LC cloud with intensities')
    plt.show()

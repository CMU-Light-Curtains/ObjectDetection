import numpy as np
import pylc

from second.dynamic.devices.device import Device

class LightCurtain(Device):
    def __init__(self, env, gt_state_device, latency=16.66):
        super(LightCurtain, self).__init__(env, capacity=1)
        self.gt_state_device = gt_state_device
        self.latency = latency
        
        # hardcoded lc_device for synthia
        self.lc_device = pylc.LCDevice(
            CAMERA_PARAMS={
                'width': 640,
                'height': 480,
                'fov': 39.32012056540195,
                'matrix': np.array([[895.6921997070312, 0.0              , 320.0],
                                    [0.0              , 895.6921997070312, 240.0],
                                    [0.0              , 0.0              , 1.0  ]], dtype=np.float32),
                'distortion': [0, 0, 0, 0, 0]
            },
            LASER_PARAMS={
                'y': -0.2  # place laser 3m to the right of camera
            }
        )
        self._MAX_DEPTH = 80.0  # only consider points within this depth

    def service(self, design_pts):
        """
        Args:
            design_pts : (np.ndarray, dtype=float32, shape=(N, 2)) design points for light curtain.
                         Axis 1 corresponds to x, z in LC camera frame.
        Publishes:
            {
                "lc_image": (np.ndarray, dtype=np.float32, shape=(H, W, 4)) lc image.
                            Axis 3 corresponds to (x, y, z, i):
                                    - x : x in cam frame.
                                    - y : y in cam frame.
                                    - z : z in cam frame.
                                    - i : intensity of LC cloud, lying in [0, 255].
                "lc_cloud": (np.ndarray, dtype=np.float32, shape=(N, 4)) lc cloud.
                            Axis 2 corresponds to (x, y, z, i):
                                    - x : x in velo frame.
                                    - y : y in velo frame.
                                    - z : z in velo frame.
                                    - i : intensity of LC cloud, lying in [0, 1].
            }
        """
        yield self.env.timeout(self.latency)

        # get latest depth image at the time of publication
        if len(self.gt_state_device.stream) == 0:
                raise Exception("Light Curatin Device: gt_state_device stream empty at the time of LC publication!")
        depth_image = self.gt_state_device.stream[-1].data["depth"]  # (H, W)

        lc_image = self.lc_device.get_return(depth_image, design_pts)  # (H, W, 4)
        lc_cloud = lc_image.reshape(-1, 4)  # (N, 4)
        # Remove points which are NaNs.
        non_nan_mask = np.all(np.isfinite(lc_cloud), axis=1)
        lc_cloud = lc_cloud[non_nan_mask]  # (N, 4)
        # Convert lc_cloud to velo frame.
        lc_cloud_xyz1 = np.hstack((lc_cloud[:, :3], np.ones([len(lc_cloud), 1], dtype=np.float32)))
        lc_cloud_xyz1 = lc_cloud_xyz1 @ self.lc_device.TRANSFORMS["cTw"].T
        lc_cloud[:, :3] = lc_cloud_xyz1[:, :3]  # (N, 4)
        # Rescale LC return to [0, 1].
        lc_cloud[:, 3] /= 255.  # (N, 4)

        data = dict(lc_image=lc_image, lc_cloud=lc_cloud)
        self.publish(data)

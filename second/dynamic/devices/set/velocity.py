import matplotlib.pyplot as plt
import numpy as np

from second.dynamic.devices.device import Device
from planner import PlannerRT

class SETVelocity(Device):
    def __init__(self, env, light_curtain, latency=10, debug=False):
        super(SETVelocity, self).__init__(env, capacity=1)
        self.latency = latency
        self._debug = debug

        self.light_curtain = light_curtain

        # options
        self._PAIR_SEPARATION     = 0.5  # 0.5m apart
        self._MAX_RANGE           = 20
        self._NODES_PER_RAY       = 120  # 0.16m apart
        self._EXPANSION           = 0.3
        self._CONTRACTION         = 0.4
        self._SMOOTHNESS          = 0.05
        self._GROUND_Y            = 1.2 # LC return below _GROUND_Y is treated as coming from the ground
        self._MAX_REWARD          = 100
        self._LC_INTENSITY_THRESH = 200

        self.ranges = np.arange(1, self._NODES_PER_RAY + 1) / self._NODES_PER_RAY * self._MAX_RANGE  # (R,)
        self.R = len(self.ranges)
        self.C = self.light_curtain.lc_device.CAMERA_PARAMS["width"]  # number of camera rays
        self.planner = PlannerRT(self.light_curtain.lc_device, self.ranges, self.C)
        
        # each curtain is represented as range per camera ray
        self.frnt_curtain = np.ones([self.C], dtype=np.float32)  # (C,) initial front frontier is at 1m
        self.rear_curtain = self.frnt_curtain - self._PAIR_SEPARATION  # (C,) inital rear curtain
    
    def _update_design_pts_from_lc_image(self, lc_image, design_pts):
        """
        Args:
            lc_image: (np.ndarray, dtype=float32, shape=(H, W, 4))) output of LC device.
                        - Channels denote (x, y, z, intensity).
                        - Pixels that aren't a part of LC return will have NaNs in one of
                        the 4 channels.
                        - Intensity ranges from 0. to 255.
            design_pts: (np.ndarray, dtype=np.float32, shape=(W, 2)) design points that produced this lc_image.
        """
        mask_non_nan_cols = np.logical_not(np.isnan(lc_image).any(axis=(0, 2)))  # (W,)
        
        xz = lc_image[:, mask_non_nan_cols, :][:, :, [0, 2]]  # (H, W', 2)
        assert np.all(xz[[0], :, :] == xz)  # consistency along column
        xz = xz[0]  # (W', 2)

        # update design points
        design_pts[mask_non_nan_cols] = xz
    
    def _get_hits_from_lc_image(self, lc_image):
        hits = np.ones(lc_image.shape[:2], dtype=np.bool)  # (H, C)
            
        # mask out NaN values
        hits[np.isnan(lc_image).any(axis=2)] = 0  # (H, C)

        # mask out pixels below intensity threshold
        hits[lc_image[:, :, 3] < self._LC_INTENSITY_THRESH] = 0

        # mask out pixels that are below GROUND_Y (note that cam_y points downwards)
        hits[lc_image[:, :, 1] > self._GROUND_Y] = 0

        # collect hits across camera columns
        hits = hits.any(axis=0)  # (C,)

        return hits

    def process(self):
        while True:
            ############################################################################################################
            # Compute design points for frnt curtain
            ############################################################################################################
            
            # construct umap
            safety_mask = self.ranges.reshape(-1, 1) <= self.frnt_curtain  # (R, C)
            distances = np.abs(self.ranges.reshape(-1, 1) - self.frnt_curtain)  # (R, C)
            safe_reward = self._MAX_REWARD - distances  # (R, C)
            umap = safety_mask * safe_reward  # (R, C)

            frnt_design_pts = self.planner.get_design_points(umap)  # (C, 2)
            assert frnt_design_pts.shape == (self.C, 2)

            if self._debug:
                self.planner._visualize_curtain_rt(umap, frnt_design_pts, show=False)
                new_x, new_z = frnt_design_pts[:, 0], frnt_design_pts[:, 1]  # (C,)
                thetas = np.arctan2(new_z, new_x)
                old_x, old_z = self.frnt_curtain * np.cos(thetas), self.frnt_curtain * np.sin(thetas)
                plt.plot(old_x, old_z, c='r')
                plt.ylim(0, 30)
                plt.xlim(-10, 10)
                plt.pause(1e-4)
                plt.clf()

            # update font curtain
            self.frnt_curtain = np.linalg.norm(frnt_design_pts, axis=1)  # (C,)

            ############################################################################################################
            # Compute design points for rear curtain
            ############################################################################################################
            
            self.rear_curtain = self.frnt_curtain - self._PAIR_SEPARATION  # (C,)

            # here we won't use the planner, but just the design points to be epsilon behind the front curtain
            frnt_x, frnt_z = frnt_design_pts[:, 0], frnt_design_pts[:, 1]  # (C,)
            thetas = np.arctan2(frnt_z, frnt_x)
            rear_x, rear_z = self.rear_curtain * np.cos(thetas), self.rear_curtain * np.sin(thetas)
            rear_design_pts = np.hstack([rear_x.reshape(-1, 1), rear_z.reshape(-1, 1)])

            ############################################################################################################
            # Timeout for computations
            ############################################################################################################

            yield self.env.timeout(self.latency)

            ############################################################################################################
            # Call first light curtain process and wait
            ############################################################################################################
            
            # get light curtain return from the design points
            yield self.env.process(self.light_curtain.service(frnt_design_pts))
            frnt_lc_image = self.light_curtain.stream[-1].data["lc_image"]  # (H, W, 4)
            assert frnt_lc_image.shape[1] == self.C
            self._update_design_pts_from_lc_image(frnt_lc_image, frnt_design_pts)
            self.frnt_curtain = np.linalg.norm(frnt_design_pts, axis=1)  # (C,)

            ############################################################################################################
            # Call second light curtain process and wait
            ############################################################################################################
            
            # get light curtain return from the design points
            yield self.env.process(self.light_curtain.service(rear_design_pts))
            rear_lc_image = self.light_curtain.stream[-1].data["lc_image"]  # (H, W, 4)
            assert rear_lc_image.shape[1] == self.C
            self._update_design_pts_from_lc_image(rear_lc_image, rear_design_pts)
            self.rear_curtain = np.linalg.norm(rear_design_pts, axis=1)  # (C,)

            ############################################################################################################
            # Compute hits on camera rays from lc image
            ############################################################################################################
            
            hits = self._get_hits_from_lc_image(frnt_lc_image)

            ############################################################################################################
            # Expand/contract frontier
            ############################################################################################################

            self.frnt_curtain = self.frnt_curtain + (1 - hits) * self._EXPANSION
            self.frnt_curtain = self.frnt_curtain -      hits  * self._CONTRACTION

            # Enforce smoothness
            for i in range(len(self.frnt_curtain)):
                for j in range(len(self.frnt_curtain)):
                    if self.frnt_curtain[i] > self.frnt_curtain[j]:
                        self.frnt_curtain[i] = min(self.frnt_curtain[i], self.frnt_curtain[j] + self._SMOOTHNESS * abs(i-j))
    
    def reset(self, env):
        super(SETVelocity, self).reset(env)
        self.frnt_curtain = np.ones([self.C], dtype=np.float32)  # (C,) initial front frontier is at 1m
        self.rear_curtain = self.frnt_curtain - self._PAIR_SEPARATION  # (C,) inital rear curtain

import matplotlib.pyplot as plt
import numpy as np

from second.dynamic.devices.device import Device
from planner import PlannerRT

class SETBaseline(Device):
    def __init__(self, env, light_curtain, latency=10, debug=False):
        super(SETBaseline, self).__init__(env, capacity=1)
        self.latency = latency
        self._debug = debug

        self.light_curtain = light_curtain

        # options
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
        
        # frontier is the estimated envelope, represented as range per camera ray
        self.frontier = np.ones([self.C], dtype=np.float32)  # (C,) initial frontier is 1m

    def process(self):
        while True:
            ############################################################################################################
            # Compute design points from current frontier
            ############################################################################################################
            
            # construct umap
            safety_mask = self.ranges.reshape(-1, 1) <= self.frontier  # (R, C)
            distances = np.abs(self.ranges.reshape(-1, 1) - self.frontier)  # (R, C)
            safe_reward = self._MAX_REWARD - distances  # (R, C)
            umap = safety_mask * safe_reward  # (R, C)

            design_pts = self.planner.get_design_points(umap)  # (C, 2)
            assert design_pts.shape == (self.C, 2)

            if self._debug:
                self.planner._visualize_curtain_rt(umap, design_pts, show=False)
                new_x, new_z = design_pts[:, 0], design_pts[:, 1]  # (C,)
                thetas = np.arctan2(new_z, new_x)
                old_x, old_z = self.frontier * np.cos(thetas), self.frontier * np.sin(thetas)
                plt.plot(old_x, old_z, c='r')
                plt.ylim(0, 30)
                plt.xlim(-10, 10)
                plt.pause(1e-4)
                plt.clf()

            # update frontier
            self.frontier = np.linalg.norm(design_pts, axis=1)  # (C,)

            ############################################################################################################
            # Timeout for computations
            ############################################################################################################

            yield self.env.timeout(self.latency)

            ############################################################################################################
            # Call light curtain process and wait
            ############################################################################################################
            
            # get light curtain return from the design points
            yield self.env.process(self.light_curtain.service(design_pts))
            lc_image = self.light_curtain.stream[-1].data["lc_image"]  # (H, W, 4)
            assert lc_image.shape[1] == self.C

            ############################################################################################################
            # Compute hits on camera rays from lc image
            ############################################################################################################
            
            hits = np.ones(lc_image.shape[:2], dtype=np.bool)  # (H, C)
            
            # mask out NaN values
            hits[np.isnan(lc_image).any(axis=2)] = 0  # (H, C)

            # mask out pixels below intensity threshold
            hits[lc_image[:, :, 3] < self._LC_INTENSITY_THRESH] = 0

            # mask out pixels that are below GROUND_Y (note that cam_y points downwards)
            hits[lc_image[:, :, 1] > self._GROUND_Y] = 0

            # collect hits across camera columns
            hits = hits.any(axis=0)  # (C,)

            ############################################################################################################
            # Expand/contract frontier
            ############################################################################################################

            self.frontier = self.frontier + (1 - hits) * self._EXPANSION
            self.frontier = self.frontier -      hits  * self._CONTRACTION

            # Enforce smoothness
            for i in range(len(self.frontier)):
                for j in range(len(self.frontier)):
                    if self.frontier[i] > self.frontier[j]:
                        self.frontier[i] = min(self.frontier[i], self.frontier[j] + self._SMOOTHNESS * abs(i-j))
    
    def reset(self, env):
        super(SETBaseline, self).reset(env)
        self.frontier = np.ones([self.C], dtype=np.float32)  # (C,) initial frontier is 1m

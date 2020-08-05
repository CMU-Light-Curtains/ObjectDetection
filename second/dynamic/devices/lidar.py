import numpy as np

from second.dynamic.devices.device import Device

class Lidar(Device):
    def __init__(self, env, gt_state_device, capacity=1, latency=40):
        super(Lidar, self).__init__(env, capacity)
        self.gt_state_device = gt_state_device
        self.latency = latency

        self.sb_lidar_params = {
            "center": -0.3, # relative to camera
            "thickness": 0.05,  # in meters
            "angular_res": 0.4  # in degrees
        }
    
    def process(self):
        """
        Publishes:
            sb_lidar: (np.ndarray, dtype=np.float32, shape=(N, 3)) simulated single-beam lidar, in velo frame.
        """
        self.stream.clear()
        while True:
            # get latest state to publish
            if len(self.gt_state_device.stream) == 0:
                raise Exception("Lidar device: gt_state_device stream empty at the time of lidar publication!")
            state = self.gt_state_device.stream[-1].data

            # process points to sbl points
            points = state["points"]  # (N, 3) or (N, 6)
            sbl_points = self.convert_points_to_sb_lidar(points)  # (N, 3) or (N, 6)
            if sbl_points.shape[1] > 3:
                sbl_points = sbl_points[:, :3]  # (N, 3)
            
            self.publish(sbl_points)  # first publication is at t=0
            yield self.env.timeout(self.latency)
    
    def convert_points_to_sb_lidar(self, points):
        """
        Converts a point cloud to approximate single beam lidar, positioned at a certain height.
        Args:
            points: (np.ndarray, dtype=np.float32, shape=(N, 3+)) point cloud, in velo frame.
        Returns:
            sb_lidar: (np.ndarray, dtype=np.float32, shape=(N, 3+)) simulated single-beam lidar, in velo frame.
        """
        z_center = self.sb_lidar_params["center"]
        thickness = self.sb_lidar_params["thickness"]

        # VERTICAL SPARSITY
        # Remove points outside thickness band.
        z = points[:, 2]
        keep = (z > z_center - 0.5 * thickness) & (z < z_center + 0.5 * thickness)
        points = points[keep]

        # ANGULAR SPARSITY
        # First compute horizontal angle.
        x, y = points[:, 0], points[:, 1]
        # Remember that x and y are in velo frame.
        # Also, θ is from +y axis (left to right).
        θ = np.rad2deg(np.arctan2(x, y))  # (N,) in degrees
        θ = θ / self.sb_lidar_params["angular_res"]  # in resolution units
        
        # θ_frac are in [0, 1).
        θ_int, θ_frac = np.divmod(θ, 1)  # (N,) and (N,)
        min_int, max_int = int(θ_int.min()), int(θ_int.max())

        keep = []
        for sector_min in range(min_int, max_int + 1):
            sector_max = sector_min + 1
            in_sector_inds = np.where((θ > sector_min) & (θ < sector_max))[0]
            if len(in_sector_inds) == 0:
                continue
            # Select the theta whose frac is closest to 0.5.
            θ_frac_in_sector = θ_frac[in_sector_inds]
            closest = np.abs(θ_frac_in_sector - 0.5).argmin()
            ind = in_sector_inds[closest]
            keep.append(ind)
        points = points[keep]

        return points

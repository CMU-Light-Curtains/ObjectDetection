from second.light_curtain.policy import DPOptimizedPolicy
from second.dynamic.devices.device import Device

class DpOptimizer(Device):
    def __init__(self, env, light_curtain, latency=8):
        super(DpOptimizer, self).__init__(env, capacity=1)
        self.latency = latency
        self.policy = DPOptimizedPolicy(light_curtain.lc_device)

    def service(self, confidence_map):
        """
        Args:
            confidence_map: (np.ndarray, dtype=float32, shape=(X, Z, 3)) confidence map of detector.
                            Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                            Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                            Axis 2 corresponds to (x, z, c):
                                - x : x in camera frame.
                                - z : z in camera frame.
                                - c : confidence score lying in [0, 1].
        Publishes:
            design_points: (np.ndarray, dtype=float32, shape=(N, 2)) design points for light curtain.
                           Axis 1 corresponds to x, z in LC camera frame.
        """
        design_points = self.policy.get_design_points(confidence_map)  # (N, 2)
        yield self.env.timeout(self.latency)
        self.publish(design_points)

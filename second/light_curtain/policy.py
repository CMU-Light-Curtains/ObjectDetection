import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

REGISTERED_LC_POLICY_CLASSES = {}

def register_lc_policy(cls):
    global REGISTERED_LC_POLICY_CLASSES
    name = cls.__name__
    assert name not in REGISTERED_LC_POLICY_CLASSES, f"exist class: {REGISTERED_LC_POLICY_CLASSES}"
    REGISTERED_LC_POLICY_CLASSES[name] = cls
    return cls

def get_lc_policy_class(name):
    global REGISTERED_LC_POLICY_CLASSES
    assert name in REGISTERED_LC_POLICY_CLASSES, f"available class: {REGISTERED_LC_POLICY_CLASSES}"
    return REGISTERED_LC_POLICY_CLASSES[name]

class Policy:
    """Defines light curtain placement policies"""
    def __init__(self, lc_device):
        self._lc_device = lc_device
    
    def get_design_points(self, confidence_map):
        """
        Args:
            confidence_map: (np.ndarray, dtype=float32, shape=(X, Z, 3+)) confidence map of detector.
                            Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                            Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                            Axis 2 corresponds to (x, z, c+):
                                - x  : x in camera frame.
                                - z  : z in camera frame.
                                - c+ : confidence score of various factors, lying in [0, 1].
        Returns:
            design_points: (np.ndarray, dtype=float32, shape=(N, 2)) point cloud of design points.
                        Each point (axis 1) contains (x, z) location of design point in camera frame.

        """
        raise NotImplementedError

    def confidence2entropy(self, confidence_map):
        """
        Args:
            confidence_map: (np.ndarray, dtype=float32, shape=(X, Z, 3+)) confidence map of detector.
                            Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                            Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                            Axis 2 corresponds to (x, z, c):
                                - x  : x in camera frame.
                                - z  : z in camera frame.
                                - c+ : confidence score of various factors, lying in [0, 1].
        Returns:
            entropy_map: (np.ndarray, dtype=float32, shape=(X, Z, 3)) entropy map of detector.
                             Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                             Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                             Axis 2 corresponds to (x, z, c):
                                 - x : x in camera frame.
                                 - z : z in camera frame.
                                 - e : entopy.
        """
        xz = confidence_map[:, :, :2]  # (X, Z, 2)
        p  = confidence_map[:, :, 2:]  # (X, Z, K)
        p = p.clip(1e-5, 1-1e-5)  # (X, Z, K)
        e = -p * np.log2(p) - (1-p) * np.log2(1-p)  # (X, Z, K)
        e  = e.mean(axis=2, keepdims=True)  # (X, Z, 1)
        entropy_map = np.concatenate((xz, e), axis=2)  # (X, Z, 3)
        return entropy_map

@register_lc_policy
class DPOptimizedPolicy(Policy):
    def __init__(self, lc_device, pts_per_cam_ray=80, debug=False):
        super(DPOptimizedPolicy, self).__init__(lc_device)
        self._pts_per_cam_ray = pts_per_cam_ray
        self._debug = debug

        self._laser_vel_limit = 14000  # degrees per sec
        self._laser_acc_limit = 5.0e7 # degrees per sec per sec
        self._laser_timestep = 1.5e-5  # sec
        
        self._max_Δθ_las = self._laser_vel_limit * self._laser_timestep  # degrees
        self._max_ΔΔθ_las = self._laser_acc_limit * self._laser_timestep * self._laser_timestep  # degrees
        if self._debug:
            print(f"DPOptimizedPolicy: " + colored(f"max_Δθ_las  = {self._max_Δθ_las}°", "yellow"))
            print(f"DPOptimizedPolicy: " + colored(f"max_ΔΔθ_las = {self._max_ΔΔθ_las}°", "yellow"))

        # _cam_thetas are sorted from low to high.
        self._cam_thetas = self._compute_cam_thetas()  # in radians
        
        self._graph = None
    
    def _compute_cam_thetas(self):
        """Compute thetas of the camera (in radians), one for each pixel column"""
        cam_w = self._lc_device.CAMERA_PARAMS["width"]
        cam_h = self._lc_device.CAMERA_PARAMS["height"]
        cam_matrix = self._lc_device.CAMERA_PARAMS["matrix"]
        # Coordinates in pixels.
        x_p = np.arange(cam_w).reshape(cam_w, 1) + 0.5  # (w, 1)
        y_p = np.ones([cam_w, 1], dtype=np.float32) * 0.5 * cam_h  # (w, 1)
        xy1_p = np.hstack([x_p, y_p, np.ones([cam_w, 1], dtype=np.float32)])
        # First convert pixel coordinates to camera coordinates (inverse of camera matrix).
        xy1_c = xy1_p @ np.linalg.inv(cam_matrix).T
        # Since we set y_p to height / 2, y_c should be 0.
        # The depth value of all these points is 1.
        assert np.all(np.abs(xy1_c[:, 1]) < 1e-6)
        xz_c = xy1_c[:, [0, 2]]  # (w, 2): all the z values are 1.
        thetas = np.arctan2(xz_c[:, 1], xz_c[:, 0])  # (w,), in radians
        thetas.sort()
        if self._debug:
            print(f"DPOptimizedPolicy: camera thetas: {np.rad2deg(thetas)}")
            theta_range = np.rad2deg(thetas[-1]) - np.rad2deg(thetas[0])
            cam_fov = self._lc_device.CAMERA_PARAMS["fov"]
            print(f"DPOptimizedPolicy: theta_range/camera_fov: {theta_range:.2f}°/{cam_fov:.2f}°")
        return thetas
    
    def _interpolate(self, uncertainty_map, query_x, query_z):
        """
        Computes the 1-indexed indices of the flatteded uncertainty map according to nearst-neighbor
        interpolation. This function ignores the actual uncertainty values.

        Args:
            uncertainty_map: (np.ndarray, dtype=float32, shape=(X, Z, 3)) uncertainty map of detector.
                                Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                                Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                                Axis 2 corresponds to (x, z, c):
                                    - x : x in camera frame.
                                    - z : z in camera frame.
                                    - u : uncertainty.
            query_x: (np.ndarray, dtype=float32, shape=(N)) x coordinates of query points.
            query_z: (np.ndarray, dtype=float32, shape=(N)) z coordinates of query points.
        Returns:
            query_k: (np.ndarray, dtype=float32, shape=(N,)) indi
        """
        # This function assumes that uncertainty_map[:, :, :2] form a meshgrid with evenly spaced xs and zs.
        # It also assumes that X and Z are increasing with a fixed increment.

        x, z = uncertainty_map[:, :, 0], uncertainty_map[:, :, 1]  # (H, W)
        X, Z = uncertainty_map.shape[:2]

        Δx = (x.max() - x.min()) / (X - 1)
        Δz = (z.max() - z.min()) / (Z - 1)

        # converting to pixel coordinates
        qx = np.round((query_x - x.min()) / Δx).astype(np.int)  # (N,)
        qz = np.round((query_z - z.min()) / Δz).astype(np.int)  # (N,)

        # default k-value for points outside uncertainty map is 0.
        qk = np.zeros(len(qx), dtype=np.int)  # (N,)
        in_mask = (qx >= 0) & (qx < X) & (qz >= 0) & (qz < Z)  # (N,)
        qx_in, qz_in = qx[in_mask], qz[in_mask]  # (I,)
        qk_in = qx_in * Z + qz_in + 1  # (I,), indices for inside points start with 1
        qk[in_mask] = qk_in

        return qk
    
    def _umap2ulookup(self, uncertainty_map):
        ulookup = uncertainty_map.reshape(-1, 3)  # (N, 3)
        ulookup = ulookup[:, 2]  # (N,)
        ulookup = np.hstack([np.zeros((1,), dtype=np.float32), ulookup])  # (N+1,)
        return ulookup

    def _preprocess_graph(self, uncertainty_map):
        """
        This is a preprocessing function that creates the first order optimization graph from the uncertainty map.
        It actually ignores the uncertainty values, and only looks at the (x, z) values.
        
        First order graph means that the nodes (state) at time t consists of a point at theta t.
        Edges describe the connectivity of consecutive nodes.
        
        Args:
            uncertainty_map: (np.ndarray, dtype=float32, shape=(X, Z, 3)) uncertainty map of detector.
                             Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                             Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                             Axis 2 corresponds to (x, z, c):
                                - x : x in camera frame.
                                - z : z in camera frame.
                                - u : uncertainty.
        Returns:
            graph: {
                     "umap_wh": ths width, height of the uncertainty map for which this graph is constructed,
                     "nodes": [],
                     "edges": []
                   }

        """
        graph = {"umap_wh": uncertainty_map.shape[:2]}

        if self._debug:
            im = np.flip(uncertainty_map[:, :, 2].T, axis=0)
            plt.imshow(im, cmap='gray')
            plt.title("Uncertainty")
            plt.show()

        assert len(uncertainty_map.shape) == 3 and uncertainty_map.shape[2] == 3  # (H, W, 3)

        # Create nodes of a graph.
        # Sometimes "node at theta_t" will refer to the collection of all nodes (points)
        # in the graph that lie on the camera ray theta_t.
        nodes = []

        # Add equally spaced points to nodes.
        z_max = uncertainty_map[:, :, 1].max()
        r_ray = np.linspace(3.0, z_max, self._pts_per_cam_ray)
        for i in range(len(self._cam_thetas)):
            θ_ray = self._cam_thetas[i] * np.ones([self._pts_per_cam_ray], dtype=np.float32)
            x_ray = r_ray * np.cos(θ_ray)
            z_ray = r_ray * np.sin(θ_ray)
            k_ray = self._interpolate(uncertainty_map, x_ray, z_ray)
            points = np.hstack([e.reshape(-1, 1) for e in [x_ray, z_ray, r_ray, θ_ray, k_ray]])  # (N, 5)
            nodes.append(points)

        if self._debug:
            points = np.vstack(nodes)
            x, z, k = points[:, 0], points[:, 1], points[:, 4].astype(np.int)
            ulookup = self._umap2ulookup(uncertainty_map)
            u = ulookup[k]
            plt.scatter(x, z, c=u, cmap='gray', s=1)
            plt.title("DP Graph: Camera Ray Points w/ Interpolated Uncertainties", fontsize='x-large')
            plt.show()
        
        # TODO: Figure out how to correctly remove points outside laser's FOV.
        # # Remove points from nodes that are outside laser's FOV.
        # fov_las = self._lc_device.LASER_PARAMS["fov"]
        # lθ_min, lθ_max = 90 - 0.5 * fov_las, 90 + 0.5 * fov_las
        # for i, points in enumerate(nodes):
        #     x, z, r, θ, k = points[:, 0], points[:, 1], points[:, 2], points[:, 3], points[:, 4]
        #     y, ones = np.zeros([len(x)], dtype=np.float32), np.ones([len(x)], dtype=np.float32)
        #     xyz1 = np.hstack([e.reshape(-1, 1) for e in [x, y, z, ones]])  # (N, 4)
        #     xyz1_las = xyz1 @ self._lc_device.TRANSFORMS["cTl"].T
        #     x_las, z_las = xyz1_las[:, 0], xyz1_las[:, 2]
        #     θ_las = np.arctan2(z_las, x_las)  # (N,)
        #     θ_las_deg = np.rad2deg(θ_las)
        #     keep = (θ_las_deg > lθ_min) & (θ_las_deg < lθ_max)
        #     nodes[i] = points[keep]
        

        # Create edges of graph.
        edges = []
        # For now, the t-th edge between node_t (m points) and node_{t+1} (n points) will 
        # be the following dictionary:
        # {
        #   "contraints": mxn boolean matrix, (i,j) element is True iff going from point i
        #                 in node_t to point j in node t+1 is allowed.
        #   "Δθ_las": mxn float32 matrix.
        #             Note, the primary reward to maximize at any point in the node is the sum of
        #             uncertainties of nodes. All edges have zero cost.
        #             However, if the primary rewards for going from i in node_t to some
        #             subset of points {j ∈ J} of node_{t+1} are all equal, then we want to minimize the
        #             maximum Δθ_las_t that will be suffered throughout the future trajectory.
        #             That is, you want to minimize Δθ_las given that the same amount of uncertainty is resolved.
        #             This minimax satisfies optimal substructure, and the primary and secondary problems together
        #             also satisfy optimal substructure. Hence, this is amenable to DP.
        # }

        def _laser_angle_(points):
            x, z = points[:, 0], points[:, 1]
            y, ones = np.zeros([len(x)], dtype=np.float32), np.ones([len(x)], dtype=np.float32)
            xyz1 = np.hstack([e.reshape(-1, 1) for e in [x, y, z, ones]])  # (N, 4)
            xyz1_las = xyz1 @ self._lc_device.TRANSFORMS["cTl"].T
            x_las, z_las = xyz1_las[:, 0], xyz1_las[:, 2]
            θ_las = np.rad2deg(np.arctan2(z_las, x_las))  # (N,), degrees
            return θ_las
        
        for i in range(len(nodes) - 1):
            points_prev, points_next = nodes[i], nodes[i+1]
            θ_las_prev = _laser_angle_(points_prev)  # (M,) degrees
            θ_las_next = _laser_angle_(points_next)  # (N,) degrees
            Δθ_las = θ_las_next.reshape(1, -1) - θ_las_prev.reshape(-1, 1)  # (M, N) degrees
            constraints = (Δθ_las < self._max_Δθ_las) & \
                          (Δθ_las > -self._max_Δθ_las)  # (M, N)
            edges.append({
                "constraints": constraints,  # (M, N)
                "Δθ_las": Δθ_las  # (M, N)
            })
        
        if self._debug:
            mean_connectivity = np.array([edge["constraints"].mean() for edge in edges]).mean()
            mean_connectivity = f"{mean_connectivity * 100:.2f}%"
            print(f"DPOptimizedPolicy: mean connectivity across adjacent rays: {mean_connectivity}")

        graph["nodes"] = nodes
        graph["edges"] = edges
        return graph

    def get_design_points(self, confidence_map):
        """
        NOTE: This is an old function to compute design points, using only graph, i.e. using only
        velocity constraints. This algorithm doesn't use acceleration constraints.

        Args:
            confidence_map: (np.ndarray, dtype=float32, shape=(X, Z, 3)) confidence map of detector.
                            Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                            Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                            Axis 2 corresponds to (x, z, c):
                                - x : x in camera frame.
                                - z : z in camera frame.
                                - c : confidence score lying in [0, 1].
        Returns:
            design_points: (np.ndarray, dtype=float32, shape=(N, 2)) point cloud of design points.
                        Each point (axis 1) contains (x, z) location of design point in camera frame.

        """
        uncertainty_map = self.confidence2entropy(confidence_map)  # (X, Z, 3)

        if self._graph is None or self._graph["umap_wh"] != uncertainty_map.shape[:2]:
            self._graph = self._preprocess_graph(uncertainty_map)
        
        ulookup = self._umap2ulookup(uncertainty_map)  # (X*Z+1,)
        
        # Every element of dp_solution will have
        # {
        #   "v_unc": float32 array, size=(N,); value function: (maximum) sum of uncertainties possible from this point.
        #   "v_las": float32 array, size=(N,); value function: for all paths originiating from this point AND achieving
        #            the best sum of uncertainties, this is the best (minimum) sum/max of Δθ_las suffered from this point.
        #   "next_point_index": int array, size=(N,), the index of the point in the next node lying on the best path
        #                       starting from this node. Note: this is not present in the last node.
        # }
        dp_solution = []
        
        def apply_constraints(q_values, constraint_matrix, failed_constraint_value):
            """
            Args:
                q_values: (M, N)
                constraint_matrix: (M, N)
            Returns:
                constrained_q_values: (M, N)
            """
            constrained_q_values = q_values * constraint_matrix
            constrained_q_values[np.logical_not(constraint_matrix)] = failed_constraint_value
            return constrained_q_values

        # Start solving DP from the end.
        curr_points = self._graph["nodes"][-1]
        dp_elem = {
            "v_unc": ulookup[curr_points[:, 4].astype(np.int)],
            "v_las": np.zeros([len(curr_points)], dtype=np.float32)
        }
        dp_solution.insert(0, dp_elem)

        # Backward pass to compute q values and v values.
        # There is one less edge than number of nodes.
        for t in reversed(range(len(self._graph["edges"]))):
            node = self._graph["nodes"][t]
            edge = self._graph["edges"][t]

            constraints = edge["constraints"]  # (M, N)
            Δθ_las = edge["Δθ_las"]  # (M, N), degrees

            next_v_unc = dp_solution[0]["v_unc"]  # (N,)
            next_v_las = dp_solution[0]["v_las"]  # (N,)

            # Step 1: find v_unc.
            # Find unconstrained q values for uncertainty.
            r_unc = ulookup[node[:, 4].astype(np.int)]  # (M,)
            r_unc = r_unc.reshape(-1, 1)  # (M, 1) reward in current timestep
            unconstrained_q_unc = r_unc + next_v_unc  # (M, N)
            constraints_unc = constraints
            # q_unc:
            #   * -inf, if constraints_unc is False.
            #   * unconstrained_q_unc, if constraints_unc is True.
            q_unc = apply_constraints(unconstrained_q_unc, constraints_unc, -np.inf)  # (M, N)
            v_unc = q_unc.max(axis=1)  # (M,)

            # Step 2: find v_las.
            r_las = np.abs(Δθ_las)  # (M, N) this is actually the cost
            # unconstrained_q_las = np.maximum(r_las, next_v_las)  # (M, N)
            # unconstrained_q_las = r_las + next_v_las  # (M, N)
            unconstrained_q_las = np.square(r_las) + next_v_las  # (M, N)
            # Constraints: only consider a subset of those points in next node that lie on best path for max sum unc.
            constraints_las = (q_unc == v_unc.reshape(-1, 1))  # (M, N)
            # q_las:
            #   - +inf, if constraints_las is False.
            #   - unconstrained_q_las if constraints_las is True.
            q_las = apply_constraints(unconstrained_q_las, constraints_las, np.inf)  # (M, N)
            v_las = q_las.min(axis=1)  # (M,)
            next_point_index = q_las.argmin(axis=1)  # (M,)

            if self._debug:
                assert not np.any(np.isnan(q_unc))
                assert not np.any(np.isnan(v_unc))
                assert not np.any(np.isnan(q_las))
                assert not np.any(np.isnan(v_las))

            dp_elem = {
                "v_unc": v_unc,
                "v_las": v_las,
                "next_point_index": next_point_index
            }

            dp_solution.insert(0, dp_elem)
            
        # Forward pass to compute optimal trajectory.
        optimal_trajectory = []
        # Each ot_elem is of the following:
        # {
        #   "point": float32 array, shape=(1, 5) (x, z, r, θ, k),
        #   "sum_unc": float32, optimal sum of uncertainties,
        #   "max_Δθ_las" float32, optimal Δθ_las
        # }

        # Compute point_index of first dp_elem.
        v_unc, v_las = dp_solution[0]["v_unc"], dp_solution[0]["v_las"]
        inds = np.arange(len(v_unc))
        keep = (v_unc == v_unc.max())
        v_las, inds = v_las[keep], inds[keep]
        curr_point_index = inds[np.argmin(v_las)]

        for t in range(len(dp_solution)):
            dp_elem = dp_solution[t]
            points = self._graph["nodes"][t]
            optimal_trajectory.append({
                "point": points[[curr_point_index]],
                "v*_unc": dp_elem["v_unc"][curr_point_index],
                "v*_las": dp_elem["v_las"][curr_point_index]
            })
            if t < len(dp_solution) - 1:
                curr_point_index = dp_elem["next_point_index"][curr_point_index]

        design_points = np.vstack([ot_elem["point"] for ot_elem in optimal_trajectory])  # (T, 5)
        design_points = design_points[:, :2]  # (T, 2)
        design_points = np.flip(design_points, axis=0)  # (T, 2) sort x in ascending order
        
        if self._debug:
            # Print optimal values.
            v_unc, v_las = optimal_trajectory[0]['v*_unc'], optimal_trajectory[0]['v*_las']
            print(f"DPOptimizedPolicy: " + colored(f"Optimal v_unc = {v_unc:.3f}", "red"))
            print(f"DPOptimizedPolicy: " + colored(f"Optimal v_las = {v_las:.3f}", "red"))

            flattened_umap = uncertainty_map.reshape(-1, 3)
            x, z, u = flattened_umap[:, 0], flattened_umap[:, 1], flattened_umap[:, 2]
            plt.scatter(x, z, c=u, cmap='gray')
            plt.plot(design_points[:, 0], design_points[:, 1], linewidth=1.5, c='r')
            # plt.scatter(design_points[:, 0], design_points[:, 1], s=1, c='w')
            plt.show()
        
        return design_points


class GreedyDP(DPOptimizedPolicy):
    def __init__(self, lc_device, pts_per_cam_ray=80, ties=None, debug=False):
        super(GreedyDP, self).__init__(lc_device, pts_per_cam_ray, debug)
        self._ties = ties

    def get_design_points(self, confidence_map):
        """
        NOTE: This is an old function to compute design points, using only graph, i.e. using only
        velocity constraints. This algorithm doesn't use acceleration constraints.

        Args:
            confidence_map: (np.ndarray, dtype=float32, shape=(X, Z, 3)) confidence map of detector.
                            Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                            Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                            Axis 2 corresponds to (x, z, c):
                                - x : x in camera frame.
                                - z : z in camera frame.
                                - c : confidence score lying in [0, 1].
        Returns:
            design_points: (np.ndarray, dtype=float32, shape=(N, 2)) point cloud of design points.
                        Each point (axis 1) contains (x, z) location of design point in camera frame.

        """
        uncertainty_map = self.confidence2entropy(confidence_map)  # (X, Z, 3)

        if self._graph is None or self._graph["umap_wh"] != uncertainty_map.shape[:2]:
            self._graph = self._preprocess_graph(uncertainty_map)
        
        ulookup = self._umap2ulookup(uncertainty_map)  # (X*Z+1,)
            
        # Forward pass to compute optimal trajectory.
        optimal_trajectory = []

        # First point: break ties randomly.
        points = self._graph["nodes"][0]  # (M, 5)
        r_unc = ulookup[points[:, 4].astype(np.int)]  # (M,)
        keep = np.where(r_unc == r_unc.max())[0]  # (M',)
        curr_point_index = np.random.choice(keep)
        optimal_trajectory.append({
            "point": points[[curr_point_index]],
            "v*_unc": 0,
            "v*_las": 0
        })

        for t in range(1, len(self._graph["nodes"])):
            points = self._graph["nodes"][t]  # (N, 5)

            # Edge between current and previous point.
            edge = self._graph["edges"][t-1]
            constraints = edge["constraints"]  # (M, N)
            Δθ_las = edge["Δθ_las"]  # (M, N)

            neighboor_inds = np.where(constraints[curr_point_index])[0]  # (N',)
            if len(neighboor_inds) == 0:
                # There are no valid neighbors in current timestep.
                # Light curtain cannot proceed any further
                break
            
            Δθ_las = Δθ_las[curr_point_index, neighboor_inds]  # (N')
            points = points[neighboor_inds]  # (N', 5)
            r_unc = ulookup[points[:, 4].astype(np.int)]  # (N',)

            keep = (r_unc == r_unc.max())  # (N')
            neighboor_inds = neighboor_inds[keep]  # (N'')
            Δθ_las = Δθ_las[keep]  # (N'')

            if self._ties == "random":
                curr_point_index = np.random.choice(neighboor_inds)
            elif self._ties == "minvel":
                r_las = np.abs(Δθ_las)
                curr_point_index = neighboor_inds[np.argmin(r_las)]
            else:
                raise Exception("ties must be one of random/minvel")
            
            # Add the point to the optimal trajectory.
            optimal_trajectory.append({
                "point": self._graph["nodes"][t][[curr_point_index]],
                "v*_unc": 0,
                "v*_las": 0
            })

        design_points = np.vstack([ot_elem["point"] for ot_elem in optimal_trajectory])  # (T, 5)
        design_points = design_points[:, :2]  # (T, 2)
        design_points = np.flip(design_points, axis=0)  # (T, 2) sort x in ascending order
        
        if self._debug:
            # Print optimal values.
            v_unc, v_las = optimal_trajectory[0]['v*_unc'], optimal_trajectory[0]['v*_las']
            print(f"DPOptimizedPolicy: " + colored(f"Optimal v_unc = {v_unc:.3f}", "red"))
            print(f"DPOptimizedPolicy: " + colored(f"Optimal v_las = {v_las:.3f}", "red"))

            flattened_umap = uncertainty_map.reshape(-1, 3)
            x, z, u = flattened_umap[:, 0], flattened_umap[:, 1], flattened_umap[:, 2]
            plt.scatter(x, z, c=u, cmap='hot')
            plt.plot(design_points[:, 0], design_points[:, 1], linewidth=1, c='b')
            plt.scatter(design_points[:, 0], design_points[:, 1], s=1, c='w')
            plt.show()
        
        return design_points

@register_lc_policy
class GreedyDPTiesRandom(GreedyDP):
    def __init__(self, lc_device, pts_per_cam_ray=80, debug=False):
        super(GreedyDPTiesRandom, self).__init__(lc_device, pts_per_cam_ray, "random", debug)

@register_lc_policy
class GreedyDPTiesMinVel(GreedyDP):
    def __init__(self, lc_device, pts_per_cam_ray=80, debug=False):
        super(GreedyDPTiesMinVel, self).__init__(lc_device, pts_per_cam_ray, "minvel", debug)


def _parallel_lc_extent(lc_device, z):
    """
    For a light curtain device lc_device, compute the leftmost and rightmost x
    that lie in the camera's field of view at z.
    """
    fov = lc_device.CAMERA_PARAMS["fov"]  # in degrees
    angle = np.deg2rad(0.5 * fov)  # angle between z and x
    x_max = z / np.tan(angle)
    x_min = -x_max
    return x_min, x_max


@register_lc_policy
class FixedParallel15(Policy):
    def get_design_points(self, confidence_map):
        # Places a curtain at Z = 15.
        z = 15
        x_min, x_max = _parallel_lc_extent(self._lc_device, z)
        design_pts_x = np.linspace(x_min, x_max, 1600).reshape(-1, 1)  # 1600 points
        design_pts_z = np.ones_like(design_pts_x) * z  # 15m away
        design_pts = np.hstack((design_pts_x, design_pts_z))
        return design_pts

@register_lc_policy
class FixedParallel30(Policy):
    def get_design_points(self, confidence_map):
        # Places a curtain at Z = 30.
        z = 30
        x_min, x_max = _parallel_lc_extent(self._lc_device, z)
        design_pts_x = np.linspace(x_min, x_max, 1600).reshape(-1, 1)  # 1600 points
        design_pts_z = np.ones_like(design_pts_x) * z  # 30m away
        design_pts = np.hstack((design_pts_x, design_pts_z))
        return design_pts

@register_lc_policy
class FixedParallel45(Policy):
    def get_design_points(self, confidence_map):
        # Places a curtain at Z = 45.
        z = 45
        x_min, x_max = _parallel_lc_extent(self._lc_device, z)
        design_pts_x = np.linspace(x_min, x_max, 1600).reshape(-1, 1)  # 1600 points
        design_pts_z = np.ones_like(design_pts_x) * z  # 45m away
        design_pts = np.hstack((design_pts_x, design_pts_z))
        return design_pts

@register_lc_policy
class MaxEntropy(Policy):
    def get_design_points(self, confidence_map):
        entropy_map = self.confidence2entropy(confidence_map)  # (X, Z, 3)
        xz = entropy_map[:, :, :2]  # (X, Z, 2)
        e  = entropy_map[:, :, 2]  # (X, Z)

        x_inds = np.arange(xz.shape[0])  # (X,)
        z_inds = np.argmax(e, axis=1)  # (X,)
        design_pts = xz[x_inds, z_inds]
        return design_pts

@register_lc_policy
class ParallelMaxEntropy(Policy):
    def __init__(self, lc_device, debug=False):
        super(ParallelMaxEntropy, self).__init__(lc_device)

        self._fov_mask = {'cmap_wh': None, 'mask': None}
        self._debug = debug
    
    def _create_fov_mask(self, confidence_map):
        """
        Args:
            confidence_map: (np.ndarray, dtype=float32, shape=(X, Z, 3)) confidence map of detector.
                            Axis 0 corresponds to increasing X (camera frame) / decreasing Y (velo frame).
                            Axis 1 corresponds to increasing Z (camera frame) / increasing X (velo frame).
                            Axis 2 corresponds to (x, z, c):
                                - x : x in camera frame.
                                - z : z in camera frame.
                                - c : confidence score lying in [0, 1].
        Returns:
            mask: {
                    "cmap_wh": (width, height) of confidence map.
                    "mask": np.ndarray, dtype=np.bool, shape=(X, Z)
                  }
        """
        fov_mask = {'cmap_wh': confidence_map.shape[:2]}
        x = confidence_map[:, :, 0]  # (X, Z)
        z = confidence_map[:, :, 1]  # (X, Z)
        
        θ = np.rad2deg(np.arctan2(z, x))  # (X, Z, 2), in degrees
        fov = self._lc_device.CAMERA_PARAMS["fov"]  # in degrees
        mask = (θ > 90 - 0.5 * fov) & (θ < 90 + 0.5 * fov)  # (X, Z)
        
        if self._debug:
            import matplotlib.pyplot as plt
            im = np.flip(mask.T, axis=0)
            plt.imshow(im); plt.show()
        
        fov_mask["mask"] = mask
        return fov_mask
    

    def get_design_points(self, confidence_map):
        if self._fov_mask["cmap_wh"] != confidence_map.shape[:2]:
            self._fov_mask = self._create_fov_mask(confidence_map)

        entropy_map = self.confidence2entropy(confidence_map)  # (X, Z, 3)
        xz = entropy_map[:, :, :2]  # (X, Z, 2)
        e  = entropy_map[:, :, 2]  # (X, Z)

        # Multiply entropy map by fov_mask, so that entropies outside fov aren't counted.
        e = e * self._fov_mask["mask"]  # (X, Z)

        e_z = e.sum(axis=0)  # (Z,)
        z_max_entropy = xz[0, np.argmax(e_z), 1]

        if self._debug:
            im = np.flip(e.T, axis=0)
            plt.imshow(im)
            plt.axhline(y=(e.T.shape[0] - np.argmax(e_z)),
                        color='red')
            plt.show()
        
        # LC should be at least 5m away.
        z_max_entropy = max(z_max_entropy, 5.0)

        design_pts_x = xz[:, 0, [0]]  # (X, 1)
        design_pts_z = np.ones_like(design_pts_x) * z_max_entropy  # (X, 1)
        design_pts = np.hstack((design_pts_x, design_pts_z))  # (X, 2)
        return design_pts

@register_lc_policy
class RandomParallel(Policy):
    def get_design_points(self, confidence_map):
        # Place a parallel light curtain randomly between 5m and 60m.
        z_random = np.random.uniform(5, 60)
        x_min, x_max = _parallel_lc_extent(self._lc_device, z_random)
        design_pts_x = np.linspace(x_min, x_max, 1600).reshape(-1, 1)  # 1600 points
        design_pts_z = np.ones_like(design_pts_x) * z_random
        design_pts = np.hstack((design_pts_x, design_pts_z))
        return design_pts


if __name__ == '__main__':
    import pylc
    # LC Device used in Virtual KITTI.
    lc_device = pylc.LCDevice(
            CAMERA_PARAMS={
                'width': 1242,
                'height': 375,
                'fov': 81.16352842604304,
                # This matrix given in the README is not consistent with the
                # compute_camera_instrics_matrix function.
                # 'matrix': np.array([[725,   0, 620.5],
                #                     [  0, 725, 187.0],
                #                     [  0,   0,     1]], dtype=np.float32),
                'distortion': [0, 0, 0, 0, 0]
            },
            LASER_PARAMS={
                'y': -3.0  # place laser 3m to the right of camera
            }
        )
    
    policy = DPOptimizedPolicy(lc_device, debug=True)
    confidence_map = np.load("light_curtain/uncertainty_map.npy")
    # mask_x, mask_z = np.where((confidence_map[:, :, 2] < 0.04) & \
    #                           (confidence_map[:, :, 3] < 0.066))
    # confidence_map[mask_x, mask_z, 2:] = 0
    policy.get_design_points(confidence_map)

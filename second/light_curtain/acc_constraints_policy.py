import numpy as np

class DPOptimizedPolicy(Policy):
    def __init__(self, lc_device, min_pts_per_cam_ray=80, debug=False):
        super(DPOptimizedPolicy, self).__init__(lc_device)
        self._min_pts_per_cam_ray = min_pts_per_cam_ray
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
        
        self._graphO1 = None  # graph of order 1
        self._graphO2 = None  # graph of order 2
    
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
    
    def _vis_nodes(self, nodes, stage_name):
        pts_per_cam_ray = np.array([len(points) for points in nodes])
        print(f"DPOptimizedPolicy: Statistics of {stage_name} points:")
        print(f"                   MIN    : {pts_per_cam_ray.min()}")
        print(f"                   MAX    : {pts_per_cam_ray.max()}")
        print(f"                   MEAN   : {pts_per_cam_ray.mean()}")
        print(f"                   MEDIAN : {np.median(pts_per_cam_ray)}")
        print(f"                   #ZEROS : {(pts_per_cam_ray == 0).sum()}")

        # Grid of all node points.
        all_points = np.vstack([points for points in nodes])
        plt.scatter(all_points[:, 0], all_points[:, 1], s=1)
        plt.title(f"{stage_name} Points")
        plt.show()

        # Subset of above points that are min and max distance in every camera ray.
        min_pts, max_pts = [], []
        for points in nodes:
            if len(points) == 0:
                continue
            min_r_ind = np.argmin(points[:, 2])
            max_r_ind = np.argmax(points[:, 2])
            min_pts.append(points[[min_r_ind], :])
            max_pts.append(points[[max_r_ind], :])
        min_pts, max_pts = np.vstack(min_pts), np.vstack(max_pts)
        plt.scatter(min_pts[:, 0], min_pts[:, 1], s=1, c='g', label='min dist')
        plt.scatter(max_pts[:, 0], max_pts[:, 1], s=1, c='r', label='max dist')
        plt.title(f"{stage_name} Points: Min and Max distance from Camera")
        plt.legend()
        plt.show()
    
    def _preprocess_graphO1(self, uncertainty_map):
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
            graphO1: {
                        "umap_wh": ths width, height of the uncertainty map for which this graph is constructed,
                        "nodes": [],
                        "edges": []
                      }

        """
        graphO1 = {"umap_wh": uncertainty_map.shape[:2]}

        if self._debug:
            im = np.flip(uncertainty_map[:, :, 2].T, axis=0)
            plt.imshow(im, cmap='hot')
            plt.title("Uncertainty")
            plt.show()

        assert len(uncertainty_map.shape) == 3 and uncertainty_map.shape[2] == 3
        uncertainty_map = uncertainty_map.reshape(-1, 3)  # (N, 3)
        x, z = uncertainty_map[:, 0], uncertainty_map[:, 1]  # (N,)
        # Compute r, theta with respect to camera.
        r = np.sqrt(np.square(x) + np.square(z)) # (N,)
        θ = np.arctan2(z, x)  # (N,), radians
        # "k" is the index (one-indexing!) of the point in the flattened confidence map.
        k = np.arange(len(uncertainty_map)) + 1 # (N,)

        # Throw out points that are outside camera's FOV.
        cθ_min, cθ_max = self._cam_thetas[0], self._cam_thetas[-1]
        keep = (θ > cθ_min) & (θ < cθ_max)
        x, z, r, θ, k = x[keep], z[keep], r[keep], θ[keep], k[keep]  # (N,)

        # Sort by θ in ascending order.
        inds = np.argsort(θ)
        x, z, r, θ, k = x[inds], z[inds], r[inds], θ[inds], k[inds]  # (N,)
        points = np.hstack([e.reshape(-1, 1) for e in [x, z, r, θ, k]])  # (N, 5)

        # Create nodes of a graph.
        # Sometimes "node at theta_t" will refer to the collection of all nodes (points)
        # in the graph that lie on the camera ray theta_t.
        nodes = []

        # Heatmap partition
        # A part is an array of points, one for each cθ:
        # these are the points for which cθ is the closest.
        start = 0
        for i in range(len(self._cam_thetas)):
            if i < len(self._cam_thetas) - 1:
                # Angle that lies halfway between current cθ and next cθ.
                cθ_mid = 0.5 * (self._cam_thetas[i] + self._cam_thetas[i+1])
                end = θ.searchsorted(cθ_mid)  # choose all points before cθ_mid
            else:
                end = len(self._cam_thetas)
            nodes.append(points[start:end, :])
            start = end

        # Project hmap points onto cθ's.
        for i, points in enumerate(nodes): 
            cθ = self._cam_thetas[i]
            x, z, r, θ, k = points[:, 0], points[:, 1], points[:, 2], points[:, 3], points[:, 4]  # (N,)
            Δθ = θ - cθ
            r = r * np.cos(Δθ)
            θ = np.ones(len(points), dtype=np.float32) * cθ
            x = r * np.cos(cθ)
            z = r * np.sin(cθ)
            points = np.hstack([e.reshape(-1, 1) for e in [x, z, r, θ, k]])  # (N, 5)
            nodes[i] = points

        if self._debug:
            self._vis_nodes(nodes, "Hmap")

        # Add equally spaced points to nodes. These are called "padded" points.
        # The k corresponding to padded points is 0.
        z_max = max([points[:, 1].max() for points in nodes if len(points) > 0])
        for i, hmap_points in enumerate(nodes):
            num_pad = max(self._min_pts_per_cam_ray - len(hmap_points), 0)
            r_pad = np.linspace(0., z_max, num_pad)
            θ_pad = self._cam_thetas[i] * np.ones([num_pad], dtype=np.float32)
            x_pad = r_pad * np.cos(θ_pad)
            z_pad = r_pad * np.sin(θ_pad)
            k_pad = np.zeros([num_pad], dtype=np.float32)
            pad_points = np.hstack([e.reshape(-1, 1) for e in [x_pad, z_pad, r_pad, θ_pad, k_pad]])  # (N, 5)
            nodes[i] = np.vstack([hmap_points, pad_points])
        
        if self._debug:
            self._vis_nodes(nodes, "Hmap+Pad")
        
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
        
        # if self._debug:
        #     self._vis_nodes(nodes, "Hmap+Pad+InLaserFov")

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

        graphO1["nodes"] = nodes
        graphO1["edges"] = edges
        return graphO1

    def _preprocess_graphO2(self, uncertainty_map):
        """
        This is a preprocessing function that creates the second order optimization graph from the uncertainty map.
        It actually ignores the uncertainty values, and only looks at the (x, z) values.
        
        Second order graph means that the nodes (state) at time t consists of a pair of points at theta's t and t+1.
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
            graphO2: {
                        "umap_wh": ths width, height of the uncertainty map for which this graph is constructed,
                        "nodes": [{
                                    "pointpairs": np.ndarray, dtype=int, shape=(M, 2, 6),
                                                  Axis 0: number of point pairs
                                                  Axis 1: points from prev and next graphO1 nodes
                                                  Axis 2: x, z, r, θ, k, j
                                    "Δθ_las": np.ndarray, dtype=float32, shape=(M,)
                                  }],
                        "edges": [{
                                    "constraints": np.ndarray, dtype=bool, shape=(M, N),
                                    "ΔΔθ_las": np.ndarray, dtype=float32, shape=(M, N)
                                 }]
                      }

        """
        graphO2 = {"umap_wh": uncertainty_map.shape[:2]}

        # The t-th node will be an (K, 2) int array whose two values are the indices of the points in
        # node t and node t+1 of graphO1. It will only store those pairs that are valid according to
        # the edge constraints of graphO1.
        # For T thetas, there will be a total of T-1 nodes in graphO2.
        nodes = []

        # Create nodes (T-1).
        for t in range(len(self._graphO1["edges"])):
            nodeO1_prev = self._graphO1["nodes"][t]  # (M, 5), float
            nodeO1_next = self._graphO1["nodes"][t+1]  # (N, 5), float
            edgeO1 = self._graphO1["edges"][t]
            constraintsO1 = edgeO1["constraints"]  # (M, N), bool
            Δθ_las_O1 = edgeO1["Δθ_las"]  # (M, N), float

            # j is used to denote indexing to points in nodes of graphO1.
            j_prev, j_next = np.where(constraintsO1)  # (K,) and (K,)

            points_prev = nodeO1_prev[j_prev]  # (K, 5)
            points_prev = np.hstack([points_prev, j_prev.reshape(-1, 1)])  # (K, 6)

            points_next = nodeO1_next[j_next]  # (K, 5)
            points_next = np.hstack([points_next, j_next.reshape(-1, 1)])  # (K, 6)

            pointpairs = np.stack([points_prev, points_next], axis=1)  # (K, 2, 6)
            Δθ_las = Δθ_las_O1[j_prev, j_next]  # (K,)

            nodes.append({
                "pointpairs": pointpairs,  # (K, 2, 6), int
                "Δθ_las": Δθ_las  # (K,), float
            })
        
        if self._debug:
            avg_pointpairs = np.array([len(node["pointpairs"]) for node in nodes]).mean()
            print(f"DPOptimizedPolicy: avg number of pointpairs in graphO2: {avg_pointpairs:.2f}")
        
        # The t-th edge contains transitions between the t-th and t+1-th nodes in graphO2.
        # It contains "constraints", which is an MxN boolean matrix of valid transition, similar to graphO1.
        # It also contains a matrix ΔΔθ_las corresponding to the transition.
        edges = []

        # Create edges (T-2).
        for t in range(len(nodes) - 1):
            node_prev, node_next = nodes[t], nodes[t+1]

            pointpairs_prev = node_prev["pointpairs"]  # (M, 2, 6)
            pointpairs_next = node_next["pointpairs"]  # (N, 2, 6)
            Δθ_las_prev = node_prev["Δθ_las"]  # (M,)
            Δθ_las_next = node_next["Δθ_las"]  # (N,)

            # The previous and next pairs of points must be consistent.
            c_constraints = (pointpairs_prev[:, 1, 5].reshape(-1, 1) == \
                             pointpairs_next[:, 0, 5].reshape(1, -1))  # (M, N)

            # Acceleration constraints.
            ΔΔθ_las = Δθ_las_next.reshape(1, -1) - Δθ_las_prev.reshape(-1, 1)  # (M, N)
            a_constraints = (ΔΔθ_las < self._max_ΔΔθ_las) & \
                            (ΔΔθ_las > -self._max_ΔΔθ_las)  # (M, N)

            constraints = (c_constraints & a_constraints)  # (M, N)

            edges.append({
                "constraints": constraints,  # (M, N)
                "ΔΔθ_las": ΔΔθ_las  # (M, N)
            })

        if self._debug:
            mean_connectivity = np.array([edge["constraints"].mean() for edge in edges]).mean()
            mean_connectivity = f"{mean_connectivity * 100:.2f}%"
            print(f"DPOptimizedPolicy: mean connectivity of adjacent nodes in graphO2: {mean_connectivity}")

        graphO2["nodes"] = nodes
        graphO2["edges"] = edges
        return graphO2

    
    def get_design_points_O1(self, confidence_map):
        """
        NOTE: This is an old function to compute design points, using only graphO1, i.e. using only
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
        uncertainty_map = self.confidence2entropy(confidence_map)

        if self._graphO1 is None or self._graphO1["umap_wh"] != uncertainty_map.shape[:2]:
            self._graphO1 = self._preprocess_graphO1(uncertainty_map)
        
        uncertainty_map = uncertainty_map.reshape(-1, 3)  # (N, 3)
        # Add dummy point at index 0 with 0 uncertainty.
        uncertainty_map = np.vstack([np.zeros([1, 3], dtype=np.float32), uncertainty_map])  # (N+1, 3)
        
        # Every element of dp_solution will have
        # {
        #   "v_unc": float32 array, size=(N,); value function: (maximum) sum of uncertainties possible from this point.
        #   "v_las": float32 array, size=(N,); value function: for all paths originiating from this point AND achieving
        #            the best sum of uncertainties, this is the best (minimum) sum/max of Δθ_las suffered from this point.
        #   "next_point_index": int array, size=(N,), the index of the point in the next node lying on the best path
        #                       starting from this node. Note: this is not present in the last node.
        # }
        dp_solution = []

        def points2unc(points):
            k = points[:, 4].astype(np.int)  # (M,)
            unc = uncertainty_map[k, 2]  # (M,)
            return unc
        
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
        curr_points = self._graphO1["nodes"][-1]
        dp_elem = {
            "v_unc": points2unc(curr_points),
            "v_las": np.zeros([len(curr_points)], dtype=np.float32)
        }
        dp_solution.insert(0, dp_elem)

        # Backward pass to compute q values and v values.
        # There is one less edge than number of nodes.
        for t in reversed(range(len(self._graphO1["edges"]))):
            node = self._graphO1["nodes"][t]
            edge = self._graphO1["edges"][t]

            constraints = edge["constraints"]  # (M, N)
            Δθ_las = edge["Δθ_las"]  # (M, N), degrees

            next_v_unc = dp_solution[0]["v_unc"]  # (N,)
            next_v_las = dp_solution[0]["v_las"]  # (N,)

            # Step 1: find v_unc.
            # Find unconstrained q values for uncertainty.
            r_unc = points2unc(node)  # (M,)
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
        v_unc, v_las, inds = dp_solution[0]["v_unc"], dp_solution[0]["v_las"], np.arange(len(v_unc))
        keep = (v_unc == v_unc.max())
        v_las, inds = v_las[keep], inds[keep]
        curr_point_index = inds[np.argmin(v_las)]

        for t in range(len(dp_solution)):
            dp_elem = dp_solution[t]
            points = self._graphO1["nodes"][t]
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

            uncertainty_map = uncertainty_map[1:]
            x, z, u = uncertainty_map[:, 0], uncertainty_map[:, 1], uncertainty_map[:, 2]
            plt.scatter(x, z, c=u, cmap='hot')
            plt.plot(design_points[:, 0], design_points[:, 1], linewidth=1, c='b')
            plt.scatter(design_points[:, 0], design_points[:, 1], s=1, c='w')
            plt.show()
        
            # points = np.vstack([node for node in self._graphO1["nodes"]])
            # x, z = points[:, 0], points[:, 1]
            # u = points2unc(points)
            # plt.scatter(x, z, c=u, cmap='hot')
            # plt.plot(design_points[:, 0], design_points[:, 1], linewidth=1, c='b')
            # plt.scatter(design_points[:, 0], design_points[:, 1], s=1, c='w')
            
            # for i, c0 in enumerate(self._cam_thetas):
            #     if i % 1 == 0:
            #         plt.plot([0, 70 * np.cos(c0)], [0, 70 * np.sin(c0)], c='g', linewidth=1)
            # plt.show()
        
        return design_points

    def get_design_points_O2(self, confidence_map):
        """
        This is the main function that computes design points, using graphO2 i.e. this function
        uses both velocity and acceleration constraints.

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
        uncertainty_map = self.confidence2entropy(confidence_map)

        if self._graphO1 is None or self._graphO1["umap_wh"] != uncertainty_map.shape[:2]:
            self._graphO1 = self._preprocess_graphO1(uncertainty_map)
        
        if self._graphO2 is None or self._graphO2["umap_wh"] != uncertainty_map.shape[:2]:
            self._graphO2 = self._preprocess_graphO2(uncertainty_map)
        
        uncertainty_map = uncertainty_map.reshape(-1, 3)  # (N, 3)
        # Add dummy point at index 0 with 0 uncertainty.
        uncertainty_map = np.vstack([np.zeros([1, 3], dtype=np.float32), uncertainty_map])  # (N+1, 3)
        
        # Every element of dp_solution will have
        # {
        #   "v_unc": float32 array, size=(N,); value function: (maximum) sum of uncertainties possible from this
        #            pair of points.
        #   "v_las": float32 array, size=(N,); value function: for all paths originating from this pair of points
        #            AND achieving the best sum of uncertainties, this is the best (minimum) sum/max of ΔΔθ_las
        #            suffered starting from this pair of points.
        #   "next_pointpair_index": int array, size=(N,), the index of the pair of points in the next node lying
        #                           on the best path starting from this pair of points.
        #                           Note: this is not present in the last node.
        # }
        dp_solution = []

        def points2unc(points):
            """
            Args:
                point: np.ndarray, dtype=int, shape=(K, 6) {x, z, r, θ, k, j}
            """
            k = points[:, 4].astype(np.int)  # (K,)
            unc = uncertainty_map[k, 2]  # (K,)
            return unc
        
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
        curr_pointpairs = self._graphO2["nodes"][-1]["pointpairs"]
        dp_elem = {
            "v_unc": points2unc(curr_pointpairs[:, 0, :]) + points2unc(curr_pointpairs[:, 1, :]),
            "v_las": np.zeros([len(curr_pointpairs)], dtype=np.float32)
        }
        dp_solution.insert(0, dp_elem)

        # Backward pass to compute q values and v values.
        # Note that there is one less edge than number of nodes.
        for t in reversed(range(len(self._graphO2["edges"]))):
            node = self._graphO2["nodes"][t]
            edge = self._graphO2["edges"][t]

            pointpairs = node["pointpairs"]  # (M, 2, 6)
            constraints = edge["constraints"]  # (M, N)
            ΔΔθ_las = edge["ΔΔθ_las"]  # (M, N)

            next_v_unc = dp_solution[0]["v_unc"]  # (N,)
            next_v_las = dp_solution[0]["v_las"]  # (N,)

            # Step 1: find v_unc.
            # Find unconstrained q values for uncertainty.
            r_unc = points2unc(pointpairs[:, 0, :])  # (M,) -- uncertainty of the prev points
            r_unc = r_unc.reshape(-1, 1)  # (M, 1) reward in current timestep
            unconstrained_q_unc = r_unc + next_v_unc  # (M, N)
            constraints_unc = constraints
            # q_unc:
            #   * -inf, if constraints_unc is False.
            #   * unconstrained_q_unc, if constraints_unc is True.
            q_unc = apply_constraints(unconstrained_q_unc, constraints_unc, -np.inf)  # (M, N)
            v_unc = q_unc.max(axis=1)  # (M,)

            # Step 2: find v_las.
            r_las = np.abs(ΔΔθ_las)  # (M, N) this is actually the cost
            unconstrained_q_las = np.maximum(r_las, next_v_las)  # (M, N)
            # unconstrained_q_las = r_las + next_v_las  # (M, N)
            # unconstrained_q_las = np.square(r_las) + next_v_las  # (M, N)
            # Constraints: only consider a subset of those points in next node that lie on best path for max sum unc.
            constraints_las = (q_unc == v_unc.reshape(-1, 1))  # (M, N)
            # q_las:
            #   - +inf, if constraints_las is False.
            #   - unconstrained_q_las if constraints_las is True.
            q_las = apply_constraints(unconstrained_q_las, constraints_las, np.inf)  # (M, N)
            v_las = q_las.min(axis=1)  # (M,)
            next_pointpair_index = q_las.argmin(axis=1)  # (M,)

            if self._debug:
                assert not np.any(np.isnan(q_unc))
                assert not np.any(np.isnan(v_unc))
                assert not np.any(np.isnan(q_las))
                assert not np.any(np.isnan(v_las))

            dp_elem = {
                "v_unc": v_unc,  # (K,)
                "v_las": v_las,  # (K,)
                "next_pointpair_index": next_pointpair_index  # (K,)
            }

            dp_solution.insert(0, dp_elem)
            
        # Forward pass to compute optimal trajectory.
        optimal_trajectory = []
        # Each ot_elem is of the following:
        # {
        #   "pointpair": float32 array, shape=(2, 6) (x, z, r, θ, k, j),
        #   "sum_unc": float32, optimal sum of uncertainties,
        #   "max_ΔΔθ_las" float32, optimal ΔΔθ_las
        # }

        # Compute pointpair_index of first dp_elem.
        v_unc, v_las, inds = dp_solution[0]["v_unc"], dp_solution[0]["v_las"], np.arange(len(v_unc))
        keep = (v_unc == v_unc.max())
        v_las, inds = v_las[keep], inds[keep]
        curr_pointpair_index = inds[np.argmin(v_las)]

        for t in range(len(dp_solution)):
            dp_elem = dp_solution[t]
            pointpairs = self._graphO2["nodes"][t]["pointpairs"]
            optimal_trajectory.append({
                "pointpair": pointpairs[curr_pointpair_index],  # (2, 6)
                "v*_unc": dp_elem["v_unc"][curr_pointpair_index],
                "v*_las": dp_elem["v_las"][curr_pointpair_index]
            })
            if t < len(dp_solution) - 1:
                curr_pointpair_index = dp_elem["next_pointpair_index"][curr_pointpair_index]

        design_points = np.vstack([ot_elem["pointpair"][[0]] for ot_elem in optimal_trajectory])  # (T-1, 6)
        design_points = np.vstack([design_points, optimal_trajectory[-1]["pointpair"][[1]]])  # (T, 6)
        design_points = design_points[:, :2]  # (T, 2)
        design_points = np.flip(design_points, axis=0)  # (T, 2) sort x in ascending order
        
        if self._debug:
            print("DPOptimizedPolicy: optimal values")
            print("DPOptimizedPolicy: v*_unc: {:.3f}".format(optimal_trajectory[0]["v*_unc"]))
            print("DPOptimizedPolicy: v*_las: {:.5f}".format(optimal_trajectory[0]["v*_las"]))

            def _laser_angle_(points):
                x, z = points[:, 0], points[:, 1]
                y, ones = np.zeros([len(x)], dtype=np.float32), np.ones([len(x)], dtype=np.float32)
                xyz1 = np.hstack([e.reshape(-1, 1) for e in [x, y, z, ones]])  # (N, 4)
                xyz1_las = xyz1 @ self._lc_device.TRANSFORMS["cTl"].T
                x_las, z_las = xyz1_las[:, 0], xyz1_las[:, 2]
                θ_las = np.arctan2(z_las, x_las)  # (N,)
                return np.rad2deg(θ_las)
            
            pointpairs = [ot_elem["pointpair"] for ot_elem in optimal_trajectory]
            for t in range(len(pointpairs) - 1):
                assert np.all(pointpairs[t][1] == pointpairs[t+1][0])

            points = np.vstack([pointpair[[0]] for pointpair in pointpairs])
            points = np.vstack([points, pointpairs[-1][[1]]])
            θ_las = _laser_angle_(points)
            Δθ_las = θ_las[1:] - θ_las[:-1]
            ΔΔθ_las = Δθ_las[1:] - Δθ_las[:-1]
            print(f"\nDPOptimizedPolicy: Max Δθ: {np.abs(Δθ_las).max():.5f}")
            print(f"DPOptimizedPolicy: Max ΔΔθ: {np.abs(ΔΔθ_las).max():.5f}")

            uncertainty_map = uncertainty_map[1:]
            x, z, u = uncertainty_map[:, 0], uncertainty_map[:, 1], uncertainty_map[:, 2]
            plt.scatter(x, z, c=u, cmap='hot')
            plt.plot(design_points[:, 0], design_points[:, 1], linewidth=1, c='b')
            plt.scatter(design_points[:, 0], design_points[:, 1], s=1, c='w')
            plt.show()
        
            # points = np.vstack([node for node in self._graphO2["nodes"]])
            # x, z = points[:, 0], points[:, 1]
            # u = points2unc(points)
            # plt.scatter(x, z, c=u, cmap='hot')
            # plt.plot(design_points[:, 0], design_points[:, 1], linewidth=1, c='b')
            # plt.scatter(design_points[:, 0], design_points[:, 1], s=1, c='w')
            
            # for i, c0 in enumerate(self._cam_thetas):
            #     if i % 1 == 0:
            #         plt.plot([0, 70 * np.cos(c0)], [0, 70 * np.sin(c0)], c='g', linewidth=1)
            # plt.show()
        
        return design_points
    
    def get_design_points(self, confidence_map):
        return self.get_design_points_O1(confidence_map)

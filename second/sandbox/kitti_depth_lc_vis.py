from second.data.kitti_dataset import KittiDataset
import base64
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import io
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
kitti_dataset = KittiDataset(root_path='/home/sancha/data/kitti_detection',
                             info_path='/home/sancha/data/kitti_detection/kitti_infos_train.pkl',
                            )

from pylc import compute_camera_instrics_matrix

# Display Image2.
# def display_depth(idx):

query = {
    "lidar": {
        "idx": 16
    },
    "cam": {},
    "depth": {}
}
sensor_data = kitti_dataset.get_sensor_data(query)
image_str = sensor_data['cam']['data']
img = io.BytesIO(image_str)
img = mpimg.imread(img, format='PNG')
plt.imshow(img, interpolation='nearest')
plt.show()

# point_cloud = sensor_data['lidar']['points']
# calib = sensor_data['calib']

# fov = 82
# width = 1392
# height = 512

# Camera extrinsics.
# Camera frame is essentially same frame as LIDAR's, except:
# new x = old -y
# new y = old -z
# new z = old x
# velo2lccam = np.array([[0., -1.,  0., 0.],  # x <- -y
#                        [0.,  0., -1., 0.],  # y <- -z
#                        [1.,  0.,  0., 0.],  # z <-  x
#                        [0.,  0.,  0., 1.]], dtype=np.float32)

# point_cloud_lccam = point_cloud @ velo2lccam.T
# point_cloud_lccam = point_cloud_lccam[:, :3]
# depth_map = point_cloud_to_depth_mapv2(point_cloud_lccam, fov, width, height)
# plt.imshow(depth_map); plt.show()

depth_image = sensor_data["depth"]["image"]
plt.imshow(depth_image); plt.show()

# design_points = np.load('/home/sancha/repos/second.pytorch/pylc/data/design_points.npy')
# import IPython; IPython.embed()

# Design points in camera frame.
dpx = np.arange(-20, 20, 0.025).reshape(-1, 1)  # 1600 points
dpy = np.ones_like(dpx) * 0.
dpz = np.ones_like(dpx) * 20  # 20m away
design_points_cam = np.hstack((dpx, dpy, dpz, np.ones_like(dpx)))

output_image = kitti_dataset.lc_device.get_return(depth_image, design_points_cam)

# See where intensities in image are nan.
plt.imshow(np.isnan(output_image[:, :, 0])); plt.title('x is nan here'); plt.show()
plt.imshow(np.isnan(output_image[:, :, 1])); plt.title('y is nan here'); plt.show()
plt.imshow(np.isnan(output_image[:, :, 2])); plt.title('z is nan here'); plt.show()
plt.imshow(np.isnan(output_image[:, :, 3])); plt.title('intensity is nan here'); plt.show()

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
points = output_cloud[np.random.choice(len(output_cloud), 10000)]
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3])
ax.set_xlabel('X Label'); ax.set_ylabel('Y Label'); ax.set_zlabel('Z Label')
ax.set_zlim(0, 21)
plt.title('LC cloud with intensities')
plt.show()

# Plot only those points that have intensity greater than some value.
points = output_cloud[output_cloud[:, 3] > 50]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_zlim(0, 21)
plt.title('LC return for intensities above 50')
plt.show()

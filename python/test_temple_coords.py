import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

# 1. Load the two temple images and the points from data/some_corresp.npz
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']
I1 = plt.imread('../data/im1.png')
I2 = plt.imread('../data/im2.png')

# 2. Run eight_point to compute F
M = max(I1.shape[0], I1.shape[1])
fundamental_matrix = sub.eight_point(pts1, pts2, M)
print('Fundamental Matrix:\n', fundamental_matrix)

# 3. Load points in image 1 from data/temple_coords.npz
data = np.load('../data/temple_coords.npz')
temp_pts1 = data['pts1']

# 4. Run epipolar_correspondences to get points in image 2
temp_pts2 = sub.epipolar_correspondences(I1, I2, fundamental_matrix, temp_pts1)

# 5. Compute the camera projection matrix P1
data = np.load('../data/intrinsics.npz')
K1 = data['K1']
K2 = data['K2']
E = sub.essential_matrix(fundamental_matrix, K1, K2)
print('Essential Matrix:\n', E)
P1 = K1 @ np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

# 6. Use camera2 to get 4 camera projection matrices P2
extrinsics_options = hlp.camera2(E)
P2_options = [K2 @ extrinsics_options[:,:,i] for i in range(4)]

# 7. Run triangulate using the projection matrices
pts_3d_options = []
positive_depth_count = []
for i in range(4):
    pts_3d = sub.triangulate(P1, temp_pts1, P2_options[i], temp_pts2)
    pts_3d_options.append(pts_3d)
    positive_depth_count.append((pts_3d[:,2]>0).sum())

# 8. Figure out the correct P2
P2 = P2_options[np.argmax(positive_depth_count)]
pts_3d = pts_3d_options[np.argmax(positive_depth_count)]

# 9. Scatter plot the correct 3D points
fig = plt.figure()
frame = fig.add_subplot(projection='3d')
frame.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2])
frame.set_xlabel('x')
frame.set_ylabel('y')
frame.set_zlabel('z')
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
E1 = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
E2 = extrinsics_options[:,:,np.argmax(positive_depth_count)]
np.savez('../data/extrinsics.npz', R1=E1[:,:3], R2=E2[:,:3], t1=E1[:,[3]], t2=E2[:,[3]])


# ## RESULTS:
#
# Fundamental Matrix:
#  [[ 2.52874524e-09 -5.60294317e-08 -9.27849008e-06]
#  [-1.33006796e-07  7.08991923e-10  1.12443633e-03]
#  [ 2.81490965e-05 -1.08098447e-03 -4.51123569e-03]]
#
# Essential Matrix:
#  [[ 5.84548837e-03 -1.29987069e-01 -3.39748366e-02]
#  [-3.08572889e-01  1.65079610e-03  1.65468710e+00]
#  [-5.96270630e-03 -1.67505406e+00 -1.91346162e-03]]
#
# Reprojection Error:
#  0.16250673130615992
#
# test_pose.py Error:
# Reprojection Error with clean 2D points: 1.7536966797798133e-10
# Pose Error with clean 2D points: 2.3808195952942278e-12
# Reprojection Error with noisy 2D points: 6.279401296549818
# Pose Error with noisy 2D points: 1.1084450730577375
#
# test_params.py Error:
# Intrinsic Error with clean 2D points: 1.999999999999602
# Rotation Error with clean 2D points: 2.8284271247461903
# Translation Error with clean 2D points: 3.6955240416196165
# Intrinsic Error with noisy 2D points: 2.1835237791936284
# Rotation Error with noisy 2D points: 2.82842678769114
# Translation Error with noisy 2D points: 3.669696046796539
# 
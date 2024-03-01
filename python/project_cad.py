import numpy as np
import submission as sub
import matplotlib.pyplot as plt

# write your implementation here
data = np.load('../data/pnp.npz', allow_pickle=True)
x = data['x']
X = data['X']
image = data['image']
cad = data['cad'][0, 0][0]

pose = sub.estimate_pose(x, X)
K, R, t = sub.estimate_params(pose)

X_homogenous = np.hstack([X, np.ones(X.shape[0]).reshape(-1,1)])
proj = X_homogenous @ pose.T
proj = proj[:,:2] / proj[:,[2]]

plt.imshow(image)
plt.scatter(x[:,0], x[:,1], c='green', marker='o', label='Given 2D Points')
plt.scatter(proj[:,0], proj[:,1], c='black', marker='x', label='Projected 3D Points')
plt.legend()
plt.show()

cad_rotated = cad @ R

fig = plt.figure()
frame = fig.add_subplot(projection='3d')
frame.plot(cad_rotated[:,0], cad_rotated[:,1], cad_rotated[:,2], c='magenta', linewidth=0.5, markersize = 0.2)
frame.set_xlabel('x')
frame.set_ylabel('y')
frame.set_zlabel('z')
plt.show()

cad_proj = np.hstack([cad, np.ones(cad.shape[0]).reshape(-1,1)]) @ pose.T
cad_proj = cad_proj[:,:2] / cad_proj[:,[2]]

plt.imshow(image)
plt.plot(cad_proj[:,0], cad_proj[:,1], c='magenta', linewidth=0.5, markersize = 0.2)
plt.show()
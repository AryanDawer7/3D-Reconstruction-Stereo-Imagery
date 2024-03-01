"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import matplotlib.pyplot as plt
import helper
import skimage
import scipy

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):

    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    pts1_normalized = np.hstack((pts1, np.ones((pts1.shape[0], 1)))) @ T.T
    pts2_normalized = np.hstack((pts2, np.ones((pts2.shape[0], 1)))) @ T.T

    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x1, y1, w1 = pts1_normalized[i]
        x2, y2, w2 = pts2_normalized[i]
        A[i] = [x2*x1, x2*y1, x2*w1, y2*x1, y2*y1, y2*w1, w2*x1, w2*y1, w2*w1]

    U, S, Vt = np.linalg.svd(A)
    F_normalized = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F_normalized)
    S[2] = 0
    F_rank2 = np.dot(U, np.dot(np.diag(S), Vt))

    F_refined = helper.refineF(F_rank2, pts1_normalized[:,:2], pts2_normalized[:,:2])

    F_unnorm = T.T @ F_refined @ T

    return F_unnorm

"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):

    if len(im1.shape) == 3:
        im1 = skimage.color.rgb2gray(im1)
    if len(im2.shape) == 3:
        im2 = skimage.color.rgb2gray(im2)

    window_size = 5
    pad_width = window_size // 2

    im1_padded = np.pad(im1, ((pad_width,pad_width),(pad_width,pad_width)))
    im2_padded = np.pad(im2, ((pad_width,pad_width),(pad_width,pad_width)))

    pts2 = []

    for pt1 in pts1:
        x1, y1 = pt1.astype(int)

        line_coeffs = F.dot(np.array([x1, y1, 1]))
        a, b, c = line_coeffs

        x2_values = np.arange(im2.shape[1])

        y2_values = (-a*x2_values - c) / b
        y2_values = np.round(y2_values).astype(int)

        min_dist = float('inf')
        best_pt2 = None

        patch1 = im1_padded[y1-pad_width:y1+pad_width+1, x1-pad_width:x1+pad_width+1]

        for x2, y2 in zip(x2_values, y2_values):
            if y2 - pad_width < 0 or y2 + pad_width >= im2_padded.shape[0] or \
            x2 - pad_width < 0 or x2 + pad_width >= im2_padded.shape[1]:
                continue
            patch2 = im2_padded[y2-pad_width:y2+pad_width+1, x2-pad_width:x2+pad_width+1]

            dist = np.linalg.norm(patch1 - patch2) # Using Euclidean Dist

            if dist < min_dist:
                min_dist = dist
                best_pt2 = (x2, y2 - pad_width)
        
        pts2.append(best_pt2)

    pts2 = np.array(pts2)
    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return K2.T @ F @ K1


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    pts_3d = []
    for pt1, pt2 in zip(pts1, pts2):
        x1, y1 = pt1
        x2, y2 = pt2
        A = np.array([
            y1*P1[2] - P1[1],
            P1[0] - x1*P1[2],
            y2*P2[2] - P2[1],
            P2[0] - x2*P2[2]
        ])
        u, s, vt = np.linalg.svd(A)
        pts_3d.append(vt[-1])

    pts_3d = np.array(pts_3d)

    reproj_1 = pts_3d @ P1.T
    reproj_1 = reproj_1[:,:2] / reproj_1[:,[2]]

    reproj_err = np.mean(np.linalg.norm(reproj_1-pts1, axis=1))

    print("Reprojection Error:\n", reproj_err)

    pts_3d = pts_3d[:,:3]/pts_3d[:,[3]]

    return pts_3d

"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    r2 = np.cross(R1[2, :], r1[:,0]).reshape(-1,1)
    r3 = np.cross(r2[:,0], r1[:,0]).reshape(-1,1)
    R_prime = np.hstack((r1, r2, r3)).T
    
    K1p = K2.copy()
    K2p = K2.copy()
    
    t1p = -R_prime @ c1
    t2p = -R_prime @ c2
    
    M1 = K1p @ R_prime @ np.linalg.inv(K1 @ R1)
    M2 = K2p @ R_prime @ np.linalg.inv(K2 @ R2)
    
    return M1, M2, K1p, K2p, R_prime, R_prime, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    
    def calculate_ssd(i1, i2, displacement, window_size):
        i2_shifted = np.roll(i2, -displacement, axis=1)
        diff = (i1 - i2_shifted) ** 2
        
        kernel = np.ones((window_size, window_size))
        w = (win_size - 1)//2
        
        ssd = scipy.signal.convolve2d(diff, kernel)[w:-w,w:-w]
        
        return ssd

    disparity = np.array([calculate_ssd(im1, im2, i, win_size) for i in range(max_disp+1)])
    dispM = disparity.argmin(axis=0)
    
    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = -np.linalg.inv(R1).dot(t1)
    c2 = -np.linalg.inv(R2).dot(t2)
    b = np.linalg.norm(c1 - c2)
    
    f = K1[0, 0]

    depthM = np.zeros(dispM.shape)
    non_zero_disp = dispM > 0
    depthM[non_zero_disp] = (b * f) / dispM[non_zero_disp]

    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    A = []
    for (x, y), (X, Y, Z) in zip(x, X):
        A_i = np.array([
            [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x],
            [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]
        ])
        A.append(A_i)

    A = np.concatenate(A)

    u, s, vt = np.linalg.svd(A)

    return vt[-1].reshape(3,4)


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    u, s, vt = np.linalg.svd(P)
    c = vt[-1]
    c = c[:3] / c[-1]

    K, R = scipy.linalg.rq(P[:,:3])

    t = -np.dot(R, c[:3])

    return K, R, t


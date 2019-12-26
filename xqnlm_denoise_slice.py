"""
objective :
author(s) : Ashwin de Silva
date      :
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dipy.io.gradients import read_bvals_bvecs
from utils import *
from itertools import cycle
from tqdm import tqdm


def visualize_slice(I1, I2, I3, slice):
    I1 = np.swapaxes(I1, 1, 2)
    I2 = np.swapaxes(I2, 1, 2)
    I3 = np.swapaxes(I3, 1, 2)
    fig = plt.figure(tight_layout=True)
    ax1 = fig.add_subplot('131')
    ax2 = fig.add_subplot('132')
    ax3 = fig.add_subplot('133')
    im1 = I1[:, :, slice-1].T
    im2 = I2[:, :, slice-1].T
    im3 = I3[:, :, slice-1].T
    ax1.imshow(im3, cmap='gray', origin='lower')
    ax2.imshow(im1, cmap='gray', origin='lower')
    ax3.imshow(im2, cmap='gray', origin='lower')
    ax2.set_title("Gaussian Noise")
    ax3.set_title("XQNLMs")
    ax1.set_title("Original")
    plt.show()


def weighting_function(r, h):
    return np.exp(-r**2/h**2)


def visualize_q_space(bvecs, patch=None):
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x_c = 1 * np.outer(np.cos(u), np.sin(v))
    y_c = 1 * np.outer(np.sin(u), np.sin(v))
    z_c = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_c, y_c, z_c,  rstride=4, cstride=4, color='g', linewidth=0, alpha=0.2)

    x = bvecs[:, 0]
    y = bvecs[:, 1]
    z = bvecs[:, 2]

    ax.scatter(x, y, z, c='b')

    if patch is not None:
        vec = antipodal_mapping(patch, bvecs)
        x_n = vec[:, 0]
        y_n = vec[:, 1]
        z_n = vec[:, 2]

        ax.scatter(x_n, y_n, z_n, c='r', marker='s', linewidths=3, alpha=1)

    plt.show()


def cart2sph(X):
    """
    convert cartesian coordinates to spherical coordinates
    :param X: Cartesian coordinates (An N x 3 array, N is the number of points)
    :return: Spherical coordinates (An N x 3 array)
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.pi/2 - np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    out = np.zeros(X.shape)
    out[:, 0] = r
    out[:, 1] = phi
    out[:, 2] = theta
    return out


def aep(X):
    phi_0, theta_0 = X[0, 1], X[0, 2]
    phi, theta = X[1:, 1], X[1:, 2]
    rho = np.arccos(np.sin(theta_0)*np.sin(theta) + np.cos(theta_0)*np.cos(theta)*np.cos(phi-phi_0))
    alpha = np.arctan2((np.cos(theta)*np.sin(phi-phi_0)), (np.cos(theta_0)*np.sin(theta)-np.sin(theta_0)*np.cos(theta)*np.cos(phi-phi_0)))
    out = np.zeros(X[1:, 1:].shape)
    out[:, 0] = rho
    out[:, 1] = alpha
    return out


def antipodal_mapping(patch, bvecs):
    vec = bvecs[patch, :]
    angles = [np.arccos(np.dot(bvecs[patch[0], :], v).clip(-1, 1)) for v in vec]
    i = np.where(np.array(angles) > np.pi/2)[0]
    vec[list(i), :] = -vec[list(i), :]
    angles = [np.arccos(np.dot(bvecs[patch[0], :], v).clip(-1, 1)) for v in vec]
    return vec


def pol2car(X):
    rho, alpha = X[:, 1], X[:, 2]
    x = -rho*np.cos(alpha)
    y = rho*np.sin(alpha)
    out = np.zeros(X[:, 1:].shape)
    out[:, 0] = x
    out[:, 1] = y
    return out


def H_nl(rho, alpha, n, l):
    return np.exp(1j*2*np.pi*n*(rho**2))*np.exp(1j*l*alpha)


def PCET(S, X, n, l):
    N = X.shape[0]
    rho, alpha = X[:, 0], X[:, 1]
    if len(S.shape) == 1:
        return (4./(np.pi*N))*np.dot(S, H_nl(rho, alpha, n, l))
    else:
        return (4./(np.pi*N))*np.sum(S * np.conj(H_nl(rho, alpha, n, l)), axis=1)



def extract_features(S, X, m):
    if len(S.shape) == 1:
        M = np.zeros((1, (2*order+1)**2))
    else:
        M = np.zeros((S.shape[0], (2*order+1)**2))
    idx = 0
    for n in range(-m, m+1):
        for l in range(-m, m+1):
            M[:, idx] = np.abs(PCET(S, X, n, l))
            idx += 1
    return M

# parameter
original_image = '/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm_final/data/dwis.nii.gz'
input_image = '/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm_final/data/stab.nii.gz'
sigma_file = '/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm_final/data/sigma.nii.gz'
bval_file = '/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm_final/data/bval.txt'
bvec_file = '/Users/ashwin/Semester 7/Medical_Image_Processing/paper_implementation/scripts/xq_nlm_final/data/bvec.txt'

dir = 11 # direction (between 0 and 64, 0 is for the b0 and the rest is for dwis)
slice = 27
s = 2 # search radius
r = 3 # q neighbors
order = 2 # m
beta = 0.01
W = 2*s+1

# read imagen
image = nib.load(input_image)
I = np.asarray(image.get_data(caching='unchanged'))
print('Image Shape : ', I.shape)
M = I.shape[0]
N = I.shape[1]
L = I.shape[2]

# read the orginal image
image = nib.load(original_image)
I_orig = np.asarray(image.get_data(caching='unchanged'))

# get sigma
noise = nib.load(sigma_file)
sigma = np.asarray(noise.get_data(caching='unchanged'))
variance = sigma**2
print('Sigma Shape : ', sigma.shape)

# get bvals and bvecs
bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
bvecs = np.vstack((bvecs, -bvecs))
dwis = np.where(bvals > 0)[0]
b0_loc = np.where(bvals <= 0)[0]
split_b0s_idx = cycle(b0_loc)

neighbors = angular_neighbors(bvecs, r) % I.shape[-1] # get the angular neighbors for each bvec
full_indexes = [(dwi,) + tuple(neighbors[dwi]) for dwi in range(I.shape[-1]) if dwi in dwis] # angular neighbors

print(full_indexes)

sigma = sigma[:, :, :, dir].reshape(M, N, L)
im = I[:, :, :, dir].reshape(M, N, L)
denoised_im = np.zeros((M, N, L))
original_im = I_orig[:, :, :, dir].reshape(M, N, L)

# get the angular neighbors of dir
q_neighbors = list(full_indexes[dir-1])[:]
print("q-neighbors of direction %i : %s" % (dir, str(q_neighbors)))

to_denoise = np.pad(I, ((s, s), (s, s), (s, s), (0, 0)), mode='reflect')
print("Shape after Padding : ", to_denoise.shape)

progress = list(range(M*N))

for l in range(s, L+s):
    print('\n')
    print("Slice : %i" % l)
    with tqdm(total=len(progress)) as pbar:
        for m in range(s, M+s):
            for n in range(s, N+s):
                x_search_neighborhood = to_denoise[m-s:m+s+1, n-s:n+s+1, l-s:l+s+1, :]
                S_x_q = np.zeros(((r+1)*W**3, ))
                M_patches = np.zeros(((r+1)*W**3, (2*order+1)**2))
                for i, q in enumerate(q_neighbors, start=1):
                    S_x_q[(i-1)*W**3:i*W**3] = x_search_neighborhood[..., q].flatten()
                    q_patch = list(full_indexes[q-1])[:]

                    X = aep(cart2sph(antipodal_mapping(q_patch, bvecs)))
                    S = np.zeros((W**3, r))
                    for j, q_p in enumerate(q_patch[1:], start=1):
                        S[:, j-1] = x_search_neighborhood[..., q_p].flatten()
                    M_patches[(i-1)*W**3:i*W**3, ...] = extract_features(S, X, m=order)
                S_ref = to_denoise[m, n, l, q_neighbors[1:]]
                X_ref = aep(cart2sph(antipodal_mapping(q_neighbors, bvecs)))
                M_ref = extract_features(S_ref, X_ref, m=order)
                ssd = np.sqrt(np.linalg.norm((M_ref - M_patches), ord=2, axis=1)**2)
                h = np.sqrt(2*beta*(sigma[m-s, n-s, l-s]**2)*np.linalg.norm(M_ref))
                weights = weighting_function(ssd, h)

                denoised_im[m-s, n-s, l-s] = np.dot(S_x_q, weights)/np.sum(weights)

                pbar.update(1)


visualize_slice(im, denoised_im, original_im, slice)





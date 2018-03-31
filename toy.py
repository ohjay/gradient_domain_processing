#!/usr/bin/env python36

"""
Toy reconstruction.
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, sparse, signal

def reconstruct(image):
    """Reconstruct the image using gradients.
    1. v(x + 1, y) - v(x, y) should match s(x + 1, y) - s(x, y)
    2. v(x, y + 1) - v(x, y) should match s(x, y + 1) - s(x, y)
    3. v(0, 0)               should match s(0, 0)
    """
    start_time = time.time()
    height, width = image.shape
    num_px = height * width
    num_eqns = num_px * 2 - height - width + 1

    # `A_data` will contain the -1s, then the +1s, then the final constraint
    A_data = np.ones((num_px - height) * 2 + (num_px - width) * 2 + 1)
    A_data[:A_data.size // 2] = -1

    # -1 indices
    A_i = np.arange(num_px * 2 - height - width)
    A_j = np.tile(np.arange(num_px), 2)[:-width]
    A_j = np.delete(A_j, np.arange(width - 1, num_px, width))

    # +1 indices
    A_i = np.concatenate((A_i, np.arange(num_px * 2 - height - width)))
    p1j = np.concatenate((np.arange(num_px) + 1, np.arange(num_px) + width))[:-width]
    A_j = np.concatenate((A_j, np.delete(p1j, np.arange(width - 1, num_px, width))))

    # Indices for final constraint (top left corners must be equal)
    A_i = np.concatenate((A_i, [num_px * 2 - height - width]))
    A_j = np.concatenate((A_j, [0]))

    # Overall `A` is constructed as (from top to bottom)
    # 1. x-gradient objectives
    # 2. y-gradient objectives
    # 3. Final objective
    A = sparse.coo_matrix((A_data, (A_i, A_j)), shape=(num_eqns, num_px))
    A = sparse.csc_matrix(A)

    # So `b` will be constructed in the same order
    # (...x-gradients..., ...y-gradients..., s(0, 0))
    filter_dx = np.array([[1, -1]])
    filter_dy = np.array([[1], [-1]])
    dx = signal.fftconvolve(image, filter_dx, mode='valid').flatten()
    dy = signal.fftconvolve(image, filter_dy, mode='valid').flatten()
    b = np.concatenate((dx, dy, [image[0, 0]])) # TODO final (single pixel) constraint doesn't seem to matter much; investigate

    # Solve
    v = sparse.linalg.lsqr(A, b, iter_lim=1e3)[0]
    v = np.reshape(v, (height, width))
    print('Time elapsed: %.4fs' % (time.time() - start_time))
    plt.imshow(v, cmap='gray')
    plt.show()
    misc.imsave('reconstructed.png', v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    args = parser.parse_args()
    reconstruct(misc.imread(args.image_path, flatten=True))

#!/usr/bin/env python36

"""
npr.py

Non-photorealistic rendering.

Usage:
    npr.py <image_path> [-s <float>] [-aes <float>] [-fs <int>] [-gm] [-ns]
"""

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from scipy import misc, sparse, signal
from scipy.ndimage.filters import gaussian_filter1d

def rgb2gray(rgb):
    """Source: https://stackoverflow.com/a/12201744."""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def gaussian(x, sigma=1.0):
    return np.exp(-np.square(x) / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))

def deriv_gaussian(x, sigma=1.0):
    return -x * gaussian(x, sigma) / (sigma * sigma)

def dg_filter(key, n, sigma=1.0):
    """Derivative of Gaussian filter.
    Based on https://people.eecs.berkeley.edu/~sastry/ee20/cacode.html.
    """
    x = np.arange(-n // 2, -n // 2 + n)
    y = np.arange(-n // 2, -n // 2 + n)
    xx, yy = np.meshgrid(x, y)
    if key.lower() == 'x':
        _filter = gaussian(yy, sigma) * deriv_gaussian(xx, sigma)
    elif key.lower() == 'y':
        _filter = gaussian(xx, sigma) * deriv_gaussian(yy, sigma)
    return _filter / np.linalg.norm(_filter)

def npr_stylize(image, sigma=1.0, gradient_magnitude=False,
                additive_edge_scale=-1, no_show=False, filter_size=12):
    """Painterly rendering filter using gradients.
    Based on the framework presented in https://bit.ly/2SnY6cn (PPT link).

    Estimate v, where
    1. v(x + 1, y) - v(x, y) should match [s(x + 1, y) - s(x, y)] * edge(x, y)
    2. v(x, y + 1) - v(x, y) should match [s(x, y + 1) - s(x, y)] * edge(x, y)
    3. v(x, y)               should match s(x, y)
    """
    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]  # ignore alpha channel
    assert len(image.shape) == 3 and image.shape[-1] == 3, \
        'image should be RGB [currently %r]' % (image.shape,)
    if image.dtype == np.uint8:
        image = image / 255.

    start_time = time.time()
    image_gray = rgb2gray(image)
    dx = signal.fftconvolve(image_gray, dg_filter('x', filter_size, sigma), mode='same')
    dy = signal.fftconvolve(image_gray, dg_filter('y', filter_size, sigma), mode='same')
    if gradient_magnitude:
        e = np.sqrt(dx * dx + dy * dy)
    else:
        e = canny(rgb2gray(image), sigma)
    grad_x_target = e[:, :-1] * dx[:, :-1]
    grad_y_target = e[:-1, :] * dy[:-1, :]

    # --------------------- begin A, b construction

    height, width = image.shape[:2]
    num_px = height * width
    num_eqns = num_px * 3 - height - width

    # `A_data` will contain -1s, then +1s for grads, then +1s for raw vals
    A_data = np.ones((num_px - height) * 2 + (num_px - width) * 2 + num_px)
    A_data[:(num_px * 2 - height - width)] = -1

    # -1 indices
    A_i = np.arange(num_px * 2 - height - width)
    A_j = np.tile(np.arange(num_px), 2)[:-width]
    A_j = np.delete(A_j, np.arange(width - 1, num_px, width))

    # +1 indices (gradients)
    A_i = np.concatenate((A_i, np.arange(num_px * 2 - height - width)))
    p1j = np.concatenate((np.arange(num_px) + 1, np.arange(num_px) + width))[:-width]
    A_j = np.concatenate((A_j, np.delete(p1j, np.arange(width - 1, num_px, width))))

    # +1 indices (raw values)
    _curr_row = num_px * 2 - height - width
    A_i = np.concatenate((A_i, np.arange(_curr_row, num_eqns)))
    A_j = np.concatenate((A_j, np.arange(num_px)))

    # Replicate, since three-channel data
    A_data = np.tile(A_data, 3)
    A_i = np.concatenate((A_i, A_i + num_eqns, A_i + 2 * num_eqns))
    A_j = np.concatenate((A_j, A_j + num_px, A_j + 2 * num_px))

    # Overall `A` is constructed as (from top to bottom)
    # 1. x grad objectives / 2. y grad objectives / 3. raw value objectives
    A = sparse.coo_matrix((A_data, (A_i, A_j)), shape=(num_eqns * 3, num_px * 3))
    A = sparse.csc_matrix(A)

    # So `b` will be constructed in the same order
    # (...grad x targets..., ...grad y targets..., ...image...)
    b = np.concatenate((
        grad_x_target.flatten(), grad_y_target.flatten(), image[:, :, 0].flatten(),
        grad_x_target.flatten(), grad_y_target.flatten(), image[:, :, 1].flatten(),
        grad_x_target.flatten(), grad_y_target.flatten(), image[:, :, 2].flatten()))

    # --------------------- end A, b construction

    # Solve
    result = sparse.linalg.lsqr(A, b, iter_lim=1e4)[0]
    result = np.stack((
        np.reshape(result[:num_px], (height, width)),
        np.reshape(result[num_px:2*num_px], (height, width)),
        np.reshape(result[2*num_px:], (height, width))), axis=-1)

    result += additive_edge_scale * np.expand_dims(e, -1)
    result = np.clip(result, 0.0, 1.0)
    result /= result.max()

    print('Time elapsed: %.4fs' % (time.time() - start_time))
    if not no_show:
        plt.imshow(result)
        plt.show()
    misc.imsave('images/npr_result.png', result)
    print('Saved result to `images/npr_result.png`.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, nargs='?', default='images/penguin_chick.jpg')
    parser.add_argument('--sigma', '-s', type=float, default=1.0)
    parser.add_argument('--gradient_magnitude', '-gm', action='store_true', default=False)
    parser.add_argument('--additive_edge_scale', '-aes', type=float, default=-1)
    parser.add_argument('--no_show', '-ns', action='store_true', default=False)
    parser.add_argument('--filter_size', '-fs', type=int, default=12)
    args = parser.parse_args()

    image = misc.imread(args.image)
    npr_stylize(
        image, args.sigma, args.gradient_magnitude,
        args.additive_edge_scale, args.no_show, args.filter_size)

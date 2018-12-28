#!/usr/bin/env python36

"""
blend.py

Image blending.

Usage:
    blend.py <source_path> <target_path> [-s <float>] [-mg]
"""

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import time
import argparse
import numpy as np
from scipy import misc, sparse, signal

from interface import lasso, drag_layer

def blend(source, target, location, mixed_gradients=False):
    """Blend SOURCE into TARGET. Estimate v, where
    1. v(x, y) - v(x - 1, y) should match s(x, y) - s(x - 1, y)
    2. v(x, y) - v(x + 1, y) should match s(x, y) - s(x + 1, y)
    3. v(x, y) - v(x, y - 1) should match s(x, y) - s(x, y - 1)
    4. v(x, y) - v(x, y + 1) should match s(x, y) - s(x, y + 1)
    - where if v(·, ·) is outside of the source range,
    it is replaced by the target intensity at that location.

    For example, if v(x, y + 1) is out of range,
    we should match v(x, y) to t(x, y + 1) + s(x, y) - s(x, y + 1).

    Parameters
    ----------
    source: the source image as an RGBA array (only the relevant region)
    target: the target image as an RGB array
    location: the location in TARGET of the top-left corner of SOURCE
    """
    if source.dtype == np.uint8:
        source = source / 255.
    if target.dtype == np.uint8:
        target = target / 255.
    source = source.astype(np.float32)
    target = target.astype(np.float32)

    # Mask out border of source region
    # Avoids having to do boundary checks later
    source[0,  :, -1] = 0
    source[-1, :, -1] = 0
    source[:,  0, -1] = 0
    source[:, -1, -1] = 0

    start_time = time.time()
    sh, sw = source.shape[:2]
    ly, lx = location
    num_px = sh * sw

    def ravel(y, x):
        """Convert unraveled (sh, sw) coordinates to flat index."""
        return y * sw + x

    A_data, A_i, A_j, b = [], [], [], []
    eqn_idx = 0
    for ch in range(3):
        for y in range(sh):
            for x in range(sw):
                # Check alpha channel
                if source[y, x, -1] > 0:
                    bval = 0
                    for dy, dx in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        ny = y + dy  # neighboring y
                        nx = x + dx  # neighboring x
                        source_grad = source[y, x, ch] - source[ny, nx, ch]
                        target_grad = target[ly + y, lx + x, ch] - target[ly + ny, lx + nx, ch]
                        if mixed_gradients and abs(target_grad) > abs(source_grad):
                            bval += target_grad
                        else:
                            bval += source_grad
                        if source[ny, nx, -1] > 0:
                            A_data.append(-1)
                            A_i.append(eqn_idx)
                            A_j.append(ch * num_px + ravel(ny, nx))
                        else:
                            bval += target[ly + ny, lx + nx, ch]
                    A_data.append(4)
                    A_i.append(eqn_idx)
                    A_j.append(ch * num_px + ravel(y, x))
                    b.append(bval)
                else:
                    # Not part of source, constrain to be target value
                    A_data.append(1)
                    A_i.append(eqn_idx)
                    A_j.append(ch * num_px + ravel(y, x))
                    b.append(target[ly + y, lx + x, ch])
                eqn_idx += 1  # equation for every pixel

    print('for loop | Time elapsed: %.4fs' % (time.time() - start_time))
    start_time = time.time()

    A_data = np.array(A_data)
    A_i = np.array(A_i)
    A_j = np.array(A_j)
    A = sparse.coo_matrix((A_data, (A_i, A_j)), shape=(num_px * 3, num_px * 3))
    A = sparse.csc_matrix(A)
    v = sparse.linalg.lsqr(A, b, iter_lim=1e5)[0]

    v = np.stack((
        np.reshape(v[:num_px], (sh, sw)),
        np.reshape(v[num_px:2*num_px], (sh, sw)),
        np.reshape(v[2*num_px:], (sh, sw))), axis=-1)
    target[ly:ly+sh, lx:lx+sw] = np.clip(v, 0.0, 1.0)  # paste region into target

    print('lsqr sol | Time elapsed: %.4fs' % (time.time() - start_time))
    plt.imshow(target)
    while True:
        try:
            plt.show()
        except UnicodeDecodeError:
            continue
        break
    misc.imsave('images/blended.png', target)
    print('Saved result to `images/blended.png`.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, nargs='?', default='images/penguin_chick_small.jpg')
    parser.add_argument('target', type=str, nargs='?', default='images/im1_small.jpg')
    parser.add_argument('--scale', '-s', type=float, default=1.0)  # scaling factor for the source image
    parser.add_argument('--mixed_gradients', '-mg', action='store_true', default=False)
    args = parser.parse_args()

    source = misc.imread(args.source)
    if args.scale != 1.0:
        source = misc.imresize(source, args.scale)

    source = lasso(source)
    target = misc.imread(args.target)
    sy, sx = drag_layer(source, target)
    blend(source, target, (sy, sx), args.mixed_gradients)

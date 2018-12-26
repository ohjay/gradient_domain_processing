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

    start_time = time.time()
    th, tw = target.shape[:2]
    sh, sw = source.shape[:2]
    ly, lx = location
    num_px = (sh - 2) * (sw - 2)

    def ravel(y, x):
        """Convert unraveled (sh, sw) coordinates to flat (sh - 2, sw - 2) index."""
        return (y - 1) * (sw - 2) + (x - 1)

    for ch in range(3):
        A_data, A_i, A_j, b = [], [], [], []
        eqn_idx = 0
        for y in range(1, sh - 1):
            for x in range(1, sw - 1):
                # Check alpha channel
                if source[y, x, -1] > 0:
                    Aval = 0
                    bval = 0
                    for dy, dx in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        ny = y + dy  # neighboring y
                        nx = x + dx  # neighboring x
                        if source[ny, nx, -1] > 0:
                            Aval += 1
                            # mixed gradients
                            source_grad = source[y, x, ch] - source[ny, nx, ch]
                            target_grad = target[ly + y, lx + x, ch] - target[ly + ny, lx + nx, ch]
                            if mixed_gradients and abs(target_grad) > abs(source_grad):
                                bval += target_grad
                            else:
                                bval += source_grad
                            ny2 = y + 2 * dy  # 2 away
                            nx2 = x + 2 * dx  # in gradient direction
                            if ny2 < 0 or nx2 < 0 or ny2 >= sh \
                                    or nx2 >= sw or source[ny2, nx2, -1] == 0:
                                # On edge of valid source region in grad direction
                                bval += target[ly + ny, lx + nx, ch]
                            else:
                                A_data.append(-1)
                                A_i.append(eqn_idx)
                                A_j.append(ravel(ny, nx))
                    A_data.append(Aval)
                    A_i.append(eqn_idx)
                    A_j.append(ravel(y, x))
                    b.append(bval)
                else:
                    # Not part of source, constrain to be target value
                    A_data.append(1)
                    A_i.append(eqn_idx)
                    A_j.append(ravel(y, x))
                    b.append(target[ly + y, lx + x, ch])
                eqn_idx += 1  # equation for every pixel
        A_data = np.array(A_data)
        A_i = np.array(A_i)
        A_j = np.array(A_j)
        A = sparse.coo_matrix((A_data, (A_i, A_j)), shape=(num_px, num_px))
        A = sparse.csc_matrix(A)
        v = sparse.linalg.lsqr(A, b, iter_lim=1e5)[0]
        v = np.clip(np.reshape(v, (sh - 2, sw - 2)), 0.0, 1.0)
        target[ly+1:ly+sh-1, lx+1:lx+sw-1, ch] = v  # paste region into target
    print('Time elapsed: %.4fs' % (time.time() - start_time))
    plt.imshow(target)
    plt.show()
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

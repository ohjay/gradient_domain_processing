#!/usr/bin/env python36

"""
Image editing interface.
"""

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import numpy as np

########################
# GDF SOURCE SELECTION #
########################

def path_bbox(p):
    """Return 4x2 bounding box of given path.
    Source: https://gist.github.com/lebedov/9ac425419dea5e74270db907daf49df1.
    """
    assert p.ndim == 2
    assert p.shape[1] == 2

    ix_min = p[:, 0].argmin()
    ix_max = p[:, 0].argmax()
    iy_min = p[:, 1].argmin()
    iy_max = p[:, 1].argmax()

    return np.array([[p[ix_min, 0], p[iy_min, 1]],
                     [p[ix_min, 0], p[iy_max, 1]],
                     [p[ix_max, 0], p[iy_max, 1]],
                     [p[ix_max, 0], p[iy_min, 1]]])

def lasso(image):
    """Lasso select a region in the source image.
    Adapted from https://gist.github.com/lebedov/9ac425419dea5e74270db907daf49df1.
    """
    if image.dtype == np.uint8:
        image = image / 255.

    TITLE = 'Press ENTER when satisfied with your selection.'
    fig = plt.figure()
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.set_title(TITLE)

    height, width, _ = image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pix = np.vstack((x.flatten(), y.flatten())).T
    output = None

    def onselect(verts):
        # Select elements in original array bounded by selector path.
        verts = np.array(verts)
        p = Path(verts)
        ind = p.contains_points(pix, radius=1)
        selected = np.copy(image)
        selected[:, :, 0].flat[ind] = image[:, :, 0].flat[ind] * 0.8
        selected[:, :, 1].flat[ind] = image[:, :, 1].flat[ind] * 0.8
        selected[:, :, 2].flat[ind] = image[:, :, 2].flat[ind] * 0.8

        nonlocal output
        b = path_bbox(verts)
        ymin, ymax = int(min(b[:, 1])), int(max(b[:, 1])) + 1
        xmin, xmax = int(min(b[:, 0])), int(max(b[:, 0])) + 1
        output = np.zeros_like(image)
        output[:, :, 0].flat[ind] = image[:, :, 0].flat[ind]
        output[:, :, 1].flat[ind] = image[:, :, 1].flat[ind]
        output[:, :, 2].flat[ind] = image[:, :, 2].flat[ind]
        output = output[ymin:ymax, xmin:xmax]
        alpha_mask = np.zeros((height, width))
        alpha_mask.flat[ind] = 1.0
        alpha_mask = alpha_mask[ymin:ymax, xmin:xmax]
        output = np.dstack((output, alpha_mask))

        ax.clear()
        ax.imshow(selected)
        ax.set_title(TITLE)
        ax.plot(*p.vertices.T, scalex=False, scaley=False)
        fig.canvas.draw_idle()

    def quit_figure(event):
        # Source: https://github.com/matplotlib/matplotlib/issues/830/.
        if event.key == 'enter':
            plt.close(event.canvas.figure)

    cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    lasso = LassoSelector(ax, onselect)
    plt.show()
    return output

########################
# GDF SOURCE PLACEMENT #
########################

def drag_layer(source, target):
    """Drag-and-drop interface for adding source image to target image."""
    assert source.shape[-1] == 4  # assume alpha channel
    assert target.shape[-1] == 3  # assume no alpha channel
    assert np.less(source.shape[:-1], target.shape[:-1]).all()

    if source.dtype == np.uint8:
        source = source / 255.
    if target.dtype == np.uint8:
        target = target / 255.

    th, tw = target.shape[:2]
    sh, sw = source.shape[:2]

    ey, ex = None, None  # event position
    sy, sx = 0, 0        # source position (top left)
    source_canvas = np.zeros((th, tw, 4))
    source_canvas[sy:sy+sh, sx:sx+sw] = source

    TITLE = 'Press ENTER when satisfied with your placement.'
    fig = plt.figure()
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')
    ax = fig.add_subplot(111)
    ax.imshow(target)
    ax.imshow(source_canvas)
    ax.set_title(TITLE)

    def onpress(event):
        if None in (event.ydata, event.xdata):
            return True
        nonlocal ey, ex
        ey = event.ydata
        ex = event.xdata
        return True

    def onrelease(event):
        if None in (event.ydata, event.xdata):
            return True
        nonlocal sy, sx
        sy = np.clip(sy + int(event.ydata - ey), 0, th - sh)
        sx = np.clip(sx + int(event.xdata - ex), 0, tw - sw)
        source_canvas[:] = 0
        source_canvas[sy:sy+sh, sx:sx+sw] = source

        ax.clear()
        ax.imshow(target)
        ax.imshow(source_canvas)
        ax.set_title(TITLE)
        fig.canvas.draw_idle()
        return True

    def quit_figure(event):
        # Source: https://github.com/matplotlib/matplotlib/issues/830/.
        if event.key == 'enter':
            plt.close(event.canvas.figure)

    gcf = plt.gcf()
    cid = gcf.canvas.mpl_connect('key_press_event', quit_figure)
    did = gcf.canvas.mpl_connect('button_press_event', onpress)
    eid = gcf.canvas.mpl_connect('button_release_event', onrelease)
    plt.show()
    return sy, sx

#########
# DEBUG #
#########

if __name__ == '__main__':
    from scipy import misc
    source = lasso(misc.imread('images/penguin_chick.jpg'))
    plt.imsave('source.png', source)
    print('[debug] saved lassoed region to `source.png`')

    # drag and drop
    target = misc.imread('images/im1.jpg')
    sy, sx = drag_layer(source, target)
    print('[debug] the coordinates: (%d, %d)' % (sy, sx))

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
import tkinter as tk
import PIL.Image
import PIL.ImageTk

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

def lasso(image, save=True):
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
    if save:
        plt.imsave('source.png', output)
    return output

########################
# GDF SOURCE PLACEMENT #
########################

class DragGuiTk(tk.Frame):

    MAX_CANVAS_DIM = 750

    def __init__(self, root, source_path, target_path):
        self.root = root
        tk.Frame.__init__(self, root)

        source = PIL.Image.open(source_path)
        target = PIL.Image.open(target_path)
        sw, sh = source.size
        tw, th = target.size
        _scale = DragGuiTk.MAX_CANVAS_DIM / float(tw if tw > th else th)
        source = source.resize((int(sw * _scale), int(sh * _scale)))
        target = target.resize((int(tw * _scale), int(th * _scale)), PIL.Image.ANTIALIAS)
        assert sw < tw and sh < th  # also source should have alpha channel, target should not

        self.sw = sw
        self.sh = sh
        self.tw = tw
        self.th = th
        self.scale = _scale
        self.canvas_w = int(tw * _scale)
        self.canvas_h = int(th * _scale)
        self.canvas = tk.Canvas(self, width=self.canvas_w, height=self.canvas_h)

        self.target_image = PIL.ImageTk.PhotoImage(target)
        self.source_image = PIL.ImageTk.PhotoImage(source)
        # Use `create_image` for transparency
        self.target_label = self.canvas.create_image(0, 0, image=self.target_image, anchor='nw')
        self.source_label = self.canvas.create_image(0, 0, image=self.source_image, anchor='nw')
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.onclick)
        self.canvas.bind('<ButtonRelease-1>', self.onrelease)

        self.motion_id = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.sy = 0
        self.sx = 0

    def onclick(self, event):
        self.init_x = self.sx
        self.init_y = self.sy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.motion_id = self.canvas.bind('<B1-Motion>', self.onmotion)

    def onmotion(self, event):
        _sx = self.init_x + event.x - self.drag_start_x
        _sy = self.init_y + event.y - self.drag_start_y
        self.canvas.move(self.source_label, _sx - self.sx, _sy - self.sy)
        self.sx = _sx
        self.sy = _sy

    def onrelease(self, event):
        self.canvas.unbind('<B1-Motion>', self.motion_id)

    def pp_sx(self):
        sx = int(self.sx / self.scale)
        return np.clip(sx, 0, self.tw - self.sw)

    def pp_sy(self):
        sy = int(self.sy / self.scale)
        return np.clip(sy, 0, self.th - self.sh)

def drag_layer(source_path, target_path):
    root = tk.Tk()
    root.title('Close window when satisfied with your placement.')
    root.resizable(width=False, height=False)

    gui = DragGuiTk(root, source_path, target_path)
    gui.pack()

    root.mainloop()
    return gui.pp_sy(), gui.pp_sx()

#########
# DEBUG #
#########

if __name__ == '__main__':
    from scipy import misc
    source = lasso(misc.imread('images/penguin_chick.jpg'))
    print('[debug] saved lassoed region to `source.png`')

    # drag and drop
    sy, sx = drag_layer('source.png', 'images/im1.jpg')
    print('[debug] the coordinates: (%d, %d)' % (sy, sx))

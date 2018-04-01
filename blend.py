#!/usr/bin/env python36

"""
Image blending.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, sparse, signal

def fuse(source, target, location, scale):
    """Blend SOURCE into TARGET.
    1. v(x + 1, y) - v(x, y) should match s(x + 1, y) - s(x, y)
    2. v(x, y + 1) - v(x, y) should match s(x, y + 1) - s(x, y)
    - where if v(·, ·) is outside of the source range,
    it is replaced by the target intensity at that location.

    Parameters
    ----------
    source: the source image as an RGBA array (only the relevant region)
    target: the target image as an RGB array
    location: the location in TARGET of the top-left corner of SOURCE
    scale: scaling factor for the source image. Applied before setting the location.
    """
    pass

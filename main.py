#!/usr/bin/env python36

"""
main.py

Gradient domain processing.

Usage:
    main.py <source> <target>
"""

import argparse
import numpy as np
from scipy import misc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    args = parser.parse_args()
    main(args.source, args.target)

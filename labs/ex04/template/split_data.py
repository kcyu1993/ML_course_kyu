# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio:
    # ***************************************************
    # Random shuffle the index by enumerate.
    pair = np.c_[x,y]
    np.random.shuffle(pair)
    index = np.round(x.size * ratio,0).astype('int16')
    p1, p2 = np.split(pair,[index])
    x1,y1 = zip(*p1)
    x2,y2 = zip(*p2)
    return x1,y1,x2,y2
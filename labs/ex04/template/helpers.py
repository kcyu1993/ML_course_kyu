# -*- coding: utf-8 -*-
"""some helper functions."""

import numpy as np
import os

def load_data():
    """load data."""
    current_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"dataEx3.csv")
    print(current_path)
    data = np.loadtxt(current_path, delimiter=",", skiprows=1, unpack=True)
    x = data[0]
    y = data[1]
    return x, y
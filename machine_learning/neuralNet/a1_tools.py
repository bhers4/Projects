import numpy as np

def load_defect_data():
    with np.load('a1_data.npz') as data:
        x = data['defect_x'] 
        y = data['defect_y']
    return x, y

def load_swirls():
    with np.load('a1_data.npz') as data:
        x = data['swirls_x'] 
        y = data['swirls_y']
    return x, y

def load_noisy_circles():
    with np.load('a1_data.npz') as data:
        x = data['circles_x'] 
        y = data['circles_y']
    return x, y

def load_noisy_moons():
    with np.load('a1_data.npz') as data:
        x = data['moons_x']
        y = data['moons_y']
    return x, y

def load_partitioned_circles():
    with np.load('a1_data.npz') as data:
        x = data['partitioned_circles_x']
        y = data['partitioned_circles_y']
    return x, y
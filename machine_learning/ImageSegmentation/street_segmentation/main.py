import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
from utils.loader import load_images



if __name__ == "__main__":
    img_library = "/home/ben/Documents/img_library"
    load_images(img_library)

'''
    IDEA: one way to load images just for batch is have Image.open( ) in torch dataset.__getitem__(self, i). This is a RAM efficient way
    IDEA: calculate approximate data size, along with RAM target
'''


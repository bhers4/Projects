import numpy as np 
import imageio
import os, sys
import glob

for item in glob.glob("*.ppm"):
    img = imageio.imread(item)
    imageio.imwrite(str(item.split(".")[0]+".png"), img)
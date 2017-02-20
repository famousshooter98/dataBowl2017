# pre-processing tutorial

#%matplotlib inline

import numpy as np      # Linear algebra
import pandas as pd     # data procesing
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some Constants
INPUT_FOLDER = 'E:/BE400_Data/'
#patients = os.listdir(INPUT_FOLDER)
#patients.sort()

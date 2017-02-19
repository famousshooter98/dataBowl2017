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
patients = os.listdir(INPUT_FOLDER)
patients.sort()

# Load the scans in a given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0]SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

# okay, defin a new function
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # convert to in16 (from sometimes int16)

    image = image.astype(np.int16)
    
# First pass script
# I took this from 
# https://www.kaggle.com/sentdex/data-science-bowl-2017/first-pass-through-data-w-3d-convnet/notebook

import dicom            # Reads dicom files
import os               # Directory operations
import pandas as pd     # Simple data analysis

data_dir = 'C:/Users/stephen_GAME/Google Drive/UB documents/Spring 2017/BE 400/kaggle_group/sample_images/'  # Note, this is for sample images
patients = os.listdir(data_dir)
labels_df = pd.read_csv( 'E:/BE400_Data/stage1_labels.csv', index_col=0)

labels_df.head()

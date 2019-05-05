# IMPORT REQUIRED LIBRARIES AND MODULES
import cv2
import numpy as np
import os,os.path
from hog_extractor import *
import csv
import pandas as pd
from image_processing_helpers import *
from lbp_extractor import LocalBinaryPatterns

# DEFINE REQUIRED VARIABLES
features_data = []
features_data2 = pd.DataFrame(data=None, index=None, columns=None)
data_prep_obj = ImageDataPrep()

# EXTRACT HOG FEATURES AND SAVE TO THE CSV FILE
with open('train_list.txt', 'r') as f:
	images_list = f.readlines()
	for img_path in images_list:
		final_img_path = img_path.replace('\n', '')
		print(final_img_path)
		hog_feature_vector = extract_hog_features(final_img_path)  		# Extract Hog features
		print(hog_feature_vector.shape)
		print(type(hog_feature_vector))
		#print(hog_feature_vector[0:3])
		flattenVec = hog_feature_vector.flatten('F')
		print(flattenVec.shape)
		print(type(flattenVec))
		print(flattenVec[0:3])
		mylist = flattenVec.tolist()
		print(mylist[0:3])
		# Add feature vector to the data
		#df = pd.DataFrame(hog_feature_vector, columns=None)
		features_data.append(mylist)
df = pd.DataFrame(features_data)
print(df.head)
df.to_csv('hog_features.csv', encoding='utf-8', index=False, header=False)

# EXTRACT LBP FEATURES AND SAVE TO THE CSV FILE
print('Beginning LBP Features Extraction........')
desc = LocalBinaryPatterns(16, 8)
data = []

# IMPORT REQUIRED LIBRARIES AND MODULES
import cv2
import numpy as np
import os,os.path
from hog_extractor import *
import csv
import pandas as pd
from image_processing_helpers import *
from lbp_extractor import LocalBinaryPatterns
from new_hog_extractor import HogFeatureExtractor
from datetime import datetime

startTime = datetime.now()
print(' Beginning features extraction at ' + str(startTime))
# DEFINE REQUIRED VARIABLES
train_hog_features_data = []
test_hog_features_data = []
train_lbp_features_data = []
test_lbp_features_data = []
data_prep_obj = ImageDataPrep()

# DEFINE FEATURE CONFIGURATIONS
hog_extractor_obj = HogFeatureExtractor()		# Define HOG Feature Extractor object
lbp_extractor_obj = LocalBinaryPatterns(1000, 3)  # Define LBP Feature Extractor object

# EXTRACT HOG , LBP FEATURES FOR THE TRAINING DATASET AND SAVE TO THE CSV FILE
print('Beginning Feature extraction process for training dataset.....................')
with open('train_list.txt', 'r') as f:
	training_images_list = f.readlines()
	for train_img_path in training_images_list:
		train_final_img_path = train_img_path.replace('\n', '')
		print(train_final_img_path)

		# Perform HOG Feature Extraction
		train_hog_feature_vec = hog_extractor_obj.extract_hog_features(train_final_img_path)
		print(train_hog_feature_vec.shape)
		train_hog_flatten_vec = train_hog_feature_vec.flatten('F')
		print(train_hog_flatten_vec.shape)
		print(type(train_hog_flatten_vec))
		train_hog_feat_list = train_hog_flatten_vec.tolist()
		train_hog_features_data.append(train_hog_feat_list)

		# Perform LBP Feature Extraction
		train_image = cv2.imread(train_final_img_path, cv2.IMREAD_GRAYSCALE)
		print(train_image.shape)
		train_lbp_features_vector = lbp_extractor_obj.describe(train_image)
		print(train_lbp_features_vector.shape)
		train_lbp_feat_list = train_lbp_features_vector.tolist()
		train_lbp_features_data.append(train_lbp_feat_list)

train_hog_df = pd.DataFrame(train_hog_features_data)
train_hog_df.to_csv('train_hog_features.csv', encoding='utf-8', index=False, header=False)
print('Printing the first 5 rows of the HOG features for the training data set')
print(train_hog_df.head())

train_lbp_df = pd.DataFrame(train_lbp_features_data)
train_lbp_df.to_csv('train_lbp_features.csv', encoding='utf-8', index=False, header=False)
print('Printing the first 5 rows of the LBP features for the testing data')
print(train_lbp_df.head())

print('Feature extraction process for the training dataset completed successfully......................')

print('Beginning Feature extraction process for testing dataset.....................')
# EXTRACT HOG , LBP FEATURES FOR THE TEST DATASET AND SAVE TO THE CSV FILE
with open('test_list.txt', 'r') as f:
	with open('test_list.txt', 'r') as f:
		test_images_list = f.readlines()
		for test_img_path in test_images_list:
			hog_extractor_obj = HogFeatureExtractor()  # Define HOG Feature Extractor object
			lbp_extractor_obj = LocalBinaryPatterns(1000, 3)  # Define LBP Feature Extractor object
			test_final_img_path = test_img_path.replace('\n', '')
			print(test_final_img_path)

			# Perform HOG Feature Extraction
			test_hog_feature_vec = hog_extractor_obj.extract_hog_features(test_final_img_path)
			print(test_hog_feature_vec.shape)
			test_hog_flatten_vec = test_hog_feature_vec.flatten('F')
			print(test_hog_flatten_vec.shape)
			print(type(test_hog_flatten_vec))
			test_hog_feat_list = test_hog_flatten_vec.tolist()
			test_hog_features_data.append(test_hog_feat_list)

			# Perform LBP Feature Extraction
			test_image = cv2.imread(test_final_img_path, cv2.IMREAD_GRAYSCALE)
			print(test_image.shape)
			test_lbp_features_vector = lbp_extractor_obj.describe(test_image)
			print(test_lbp_features_vector.shape)
			test_lbp_feat_list = test_lbp_features_vector.tolist()
			test_lbp_features_data.append(test_lbp_feat_list)

test_hog_df = pd.DataFrame(test_hog_features_data)
test_hog_df.to_csv('test_hog_features.csv', encoding='utf-8', index=False, header=False)
print('Printing the first 5 rows of the HOG features data')
print(test_hog_df.head())

test_lbp_df = pd.DataFrame(test_lbp_features_data)
test_lbp_df.to_csv('test_lbp_features.csv', encoding='utf-8', index=False, header=False)
print('Printing the first 5 rows of the LBP features of the test data')
print(test_lbp_df.head())

print('Feature extraction process completed successfully......................')
est_time = datetime.now() - startTime

print('It took about ' + str(est_time) + ' seconds')

# Import required libraries and modules
import cv2
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from PIL import Image


class ImageDataPrep:
    def __init__(self):
        # self.sift_object = cv2.xfeatures2d.SIFT_create()
        self.obj_class_labels = []
        self.processed_img_folder = ''  # Define the final processed image folder destination path

    # Check if image is grayscale
    def is_grey_scale(self, img_path):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        for i in range(w):
            for j in range(h):
                r, g, b = img.getpixel((i, j))
                if r != g != b: return False
        return True

    def convert_to_gray(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_img

    def perform_hist_equalization(self, image):
        # Read the image
        #img = cv2.imread(img_path)

        # Check if image is grayscale if is not, then convert to gray otherwise continue with equalization
        # if self.is_grey_scale(img_path):
        #     # Perform histogram equalization image processing
        #     equalized_img = cv2.equalizeHist(img)
        # else:
        #     gray_image = self.convert_to_gray(img)
        # Perform histogram equalization image processing
        equalized_img = cv2.equalizeHist(image)
        return equalized_img

    # Create image labels from the images list of full paths
    def create_class_labels(self, list_of_files):
        class_labels = []
        for img in list_of_files:
            class_labels.append(img.split('\\')[-2])
        return class_labels

    # This function is to save a list into a file
    def save_list(self, full_file_path, image_file_list):
        with open(full_file_path, 'w') as f:
            for item in image_file_list:
                f.write("%s\n" % item)

    def read_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise IOError("Unable to open '%s'. Ensure the image path is valid")
        else:
            return img

    # This function is to resize image with consideration to aspect ratio and interpolation type
    def resize_image(self, dim_h, dim_w, image, interpol_type=None, aspect_ratio=None):
        if aspect_ratio == 'Y':
            if image.shape[0] != dim_h and image.shape[1] != dim_w:
                r = float(dim_w) / image.shape[1]
                dim = (dim_h, int(image.shape[0] * r))
                if interpol_type.upper() == 'BICUBIC':
                    resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
                    return resized_img
                else:
                    resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            else:
                return image
        else:
           # dim = [dim_h, dim_w]
            if interpol_type.upper() == 'BICUBIC':
                resized_img = cv2.resize(image, (dim_h, dim_w), interpolation=cv2.INTER_CUBIC)
                return resized_img
            else:
                resized_img = cv2.resize(image, (dim_h, dim_w), interpolation=cv2.INTER_AREA)
                return resized_img


    def save_final_image(self, full_image_pathname, processed_img):
        print("Saving the processed image to the output folder ---------")
        # Save the image the final output folder
        cv2.imwrite(full_image_pathname, processed_img)


class ImageFeatureExtractor:
    # Initialize the object class
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SIFT_create()

    # define features
    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]

# IMPORT REQUIRED LIBRARIES AND MODULES
import cv2
from PIL import Image
import numpy as np
from image_processing_helpers import *

# A class to perform hog features extraction based on Dalal Triggs paper
class HogFeatureExtractor:
    def __init__(self, pixels_per_cell=[2, 2], gamma_val=3.5, im_size=[16, 16], stride = [2,2]):
        # USER-DEFINE REQUIRED VARIABLES
        self.pixels_per_cell = pixels_per_cell
        self.gamma_val = gamma_val  # Gamma values < 1 will shift the image towards the darker end of the spectrum while
        # gamma values > 1 will make the image appear lighter.
        # A gamma value of G=1 will have no affect on the input image
        self.bin_num = 2  # was 8
        self.im_size = im_size
        self. cells_per_block = [2, 2]  # normalising blocks of cells
        self.stride = stride  # Was 8x8 The stride, is the search window, should be equal or greater than cell size.
        # Smaller stride results in a larger number of feature vectors.
        # This is one of extremely important parameters that need to be set properly. These parameter has tremendous
        # implications on not only the accuracy of your detector, but also the speed in which your detector runs.
        self. cells_per_block = [2, 2]  # normalising blocks of cells
        # for calculating the horizontal and vertical gradients
        self.max_h = self.im_size[0]  # Affect the size of the feature vector, be set according to image hight size
        self.max_w = self.im_size[0]  # Affect the size of the feature vector, be set according to image width size
        # initialize objects
        self.data_obj = ImageDataPrep()         # object from Data Preparation class


    # Creating Image Array from numpy array to standard array - using the helper class object
    def create_array(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.data_obj.is_grey_scale(image_path) == False:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_array = np.asarray(gray_image, dtype=float)
            return image_array
        return image

    # Perform local contrast normalization and compute gradient and magnitudes
    def create_grad_array(self, image_array):
        image_array = Image.fromarray(image_array)
        if not image_array.size == self.im_size:  # image size must not be the same as the proposed size
            # use the helper class to perform resize with respect to ratio
            image_array = image_array.resize(self.im_size, resample=Image.BICUBIC)
            image_array = np.asarray(image_array, dtype=float)

        # Perform gamma correction (Power Law Transform)-Gamma correction controls the overall brightness of an image.
        # Gamma correction is a nonlinear operation used to encode and decode luminance or tristimulus values in video or still image systems
        image_array = (image_array) ** self.gamma_val

        # local contrast normalisation using mean and standard deviation
        image_array = (image_array - np.mean(image_array)) / np.std(image_array)

        # calculate the horizontal and vertical gradients (We can use sobel operator) and gradient magnitudes directions
        gradient_val = np.zeros([self.max_h, self.max_w])
        magnitude_val = np.zeros([self.max_h, self.max_w])

        for h, row in enumerate(image_array):
            for w, val in enumerate(row):
                if h - 1 >= 0 and w - 1 >= 0 and h + 1 < self.max_h and w + 1 < self.max_w:
                    dy = image_array[h + 1][w] - image_array[h - 1][w]
                    dx = row[w + 1] - row[w - 1] + 0.0001  # uses a [-1 0 1 kernel]

                    gradient_val[h][w] = np.arctan(dy / dx) * (180 / np.pi)
                    if gradient_val[h][w] < 0:
                        gradient_val[h][w] += 180

                    magnitude_val[h][w] = np.sqrt(dy * dy + dx * dx)

        return gradient_val, magnitude_val

    def calculate_histogram(self, theArray, weights):
        bins_range = (0, 180)
        bins = self.bin_num
        hist, _ = np.histogram(theArray, bins=bins, range=bins_range, weights=weights)
        return hist

    def create_hog_features(self, gradient_array, magnitude_array):
        self.max_h = int(((gradient_array.shape[0] - self.pixels_per_cell[0]) / self.stride[0]) + 1)
        self.max_w = int(((gradient_array.shape[1] - self.pixels_per_cell[1]) / self.stride[1]) + 1)
        cell_array = []
        w = 0
        h = 0
        i = 0
        j = 0

        # Creating cells
        while i < self.max_h:
            w = 0
            j = 0

            while j < self.max_w:
                for_hist = gradient_array[h:h + self.pixels_per_cell[0], w:w + self.pixels_per_cell[1]]
                for_wght = magnitude_array[h:h + self.pixels_per_cell[0], w:w + self.pixels_per_cell[1]]

                val = self.calculate_histogram(for_hist, for_wght)
                cell_array.append(val)

                j += 1
                w += self.stride[1]

            i += 1
            h += self.stride[0]

        cell_array = np.reshape(cell_array, (self.max_h, self.max_w, self.bin_num))

        # here increment is 1
        max_h = int((self.max_h - self.cells_per_block[0]) + 1)
        max_w = int((self.max_w - self.cells_per_block[1]) + 1)
        block_list = []
        w = 0
        h = 0
        i = 0
        j = 0

        while i < max_h:
            w = 0
            j = 0
            while j < max_w:
                for_norm = cell_array[h:h + self.cells_per_block[0], w:w + self.cells_per_block[1]]
                magnitude_val = np.linalg.norm(for_norm)

                arr_list = (for_norm / magnitude_val).flatten().tolist()
                block_list += arr_list
                j += 1
                w += 1
            i += 1
            h += 1

        return block_list

    # Applying hog feature extraction to the image array and return final feature vector
    def extract_hog_features(self, image_path):  # the image path must be a full path
        image_array = self.create_array(image_path)
        gradient, magnitude = self.create_grad_array(image_array)
        hog_features = self.create_hog_features(gradient, magnitude)
        hog_features = np.asarray(hog_features, dtype=float)
        hog_features = np.expand_dims(hog_features, axis=0)

        return hog_features


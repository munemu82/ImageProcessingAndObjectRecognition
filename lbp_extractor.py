# IMPORT REQUIRED LIBRARIES AND MODULES
from skimage import feature
import numpy as np

# A class to perform LBP features extraction
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):      # Need to add more important parameters if exists
        # Define number of points and radius value
        self.numPoints = numPoints
        self.radius = radius

    # Perform LBP features extraction and return its histogram
    def describe(self, image, eps=1e-7):
        # Calculate LBP features representation
        # of the image, and then use the LBP representation to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # Compute histogram normalization
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist
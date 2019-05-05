# import the necessary packages
import numpy as np
import cv2

class RootSIFT:
	def __init__(self):
		# initialize the SIFT feature extractor
		self.extractor = cv2.DescriptorExtractor_create("SIFT")

	def compute(self, image, kps, eps=1e-7):
		# compute SIFT descriptors
		(kps, descs) = self.extractor.compute(image, kps)

		# if there are no keypoints or descriptors, return an empty tuple
		if len(kps) == 0:
			return ([], None)

		# apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
		#descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)

		# return a tuple of the keypoints and descriptors
		return (kps, descs)

# it to grayscale
image = cv2.imread("image1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect Difference of Gaussian keypoints in the image
detector = cv2.FeatureDetector_create("SIFT")
kps = detector.detect(gray)

# extract normal SIFT descriptors
extractor = cv2.DescriptorExtractor_create("SIFT")
(kps, descs) = extractor.compute(gray, kps)
print ("SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape))

# extract RootSIFT descriptors
rs = RootSIFT()
(kps, descs) = rs.compute(gray, kps)
print("RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape))

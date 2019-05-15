# ImageProcessingAndObjectRecognition
A python implementation of most of my master thesis research on Object recognition and Machine learning. The project focus:
1)	Preparing image dataset  from a folder path containing sub folders each representing images of various object classes (e.g. Zebra, Kangoroo etc..) 
2)	Perform low level image processing algorithms ( Convert to grayscale, histogram equalization and resize of image per given size)
3)	Prepare training and testing sets along with image labels to be used for machine learning algorithms to perform object recognition and other tasks.
Further implementations including performing transfer learning using pre-trained deep learning models (AlexNet, and VGG).

# Usage 
1) Required library packages and modules
2) System requirements
3) Run the scripts 
 3.1)  The script is the data_prep.py:
 python data_prep.py --image_path [full path to the directory if images folders each representing a class/object] --image_final_path [full path to directory where you want to store final processed images]

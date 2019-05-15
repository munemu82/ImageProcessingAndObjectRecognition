# Import required libraries and modules
import argparse  # for defining command arguments
import os
from random import shuffle
from image_processing_helpers import *
from datetime import datetime

startTime = datetime.now()

# Define required variables
all_images_list = []
train_images_list = []
test_images_list = []
train_images_labels = []
test_images_labels = []
number_of_classes = 0
train_ratio = 0.8
img_h_size = 128
img_w_size = 128
class_names = []

# Initialize and setting up command arguments# parse cmd args
parser = argparse.ArgumentParser(
    description=" Beginning data preparation"
)
# Add command prompt argument variables
parser.add_argument('--image_path', action="store", dest="image_path", required=True)  # Must be full path
parser.add_argument('--image_final_path', action="store", dest="image_final_path", required=True)  # Must be a full path
args = vars(parser.parse_args())

# Inialize class objects
data_prep_obj = ImageDataPrep()

# Get the user input paths from command prompt
image_path = args['image_path']
final_processed_path = args['image_final_path']
if os.path.isdir(image_path) == True or os.path.isdir(os.getcwd() + "/" + image_path) == True:
    print("Commencing data preparation for the images directory " + image_path)
    print("--------------------------------------------------------------------")
    for path, subdirs, files in os.walk(image_path):
        for subdir in subdirs:  # navigate through each sub directory ( image class)
            # print(os.path.join(path, name))
            print("Getting images list from folder " + subdir)
            print("--------------------------------------------------------------------")
            os.chdir(image_path)
            # print(os.path.abspath(subdir))
            obj_dir_path = os.getcwd() + '\\' + subdir
            print(obj_dir_path)

            # Create object/class specific folder for under the processed folder
            class_folder_path = final_processed_path + '\\' + obj_dir_path.split('\\')[-1]
            try:
                os.mkdir(class_folder_path)
            except OSError:
                print("Creation of the directory %s failed" % class_folder_path)
            else:
                print("Successfully created the directory %s " % class_folder_path)

            # add the object to the label
            count = 1
            for file in os.listdir(obj_dir_path):
                # print(os.path.abspath(subdir+'\\'+file))
                class_label = os.path.abspath(subdir + '\\' + file).split('\\')[-2]

                # Add the class label to the list of class names
                class_names.append(class_label)

                # PERFORM IMAGE PROCESSING
                print(os.path.abspath(subdir + '\\' + file))
                the_image = data_prep_obj.read_image(os.path.abspath(subdir + '\\' + file))  # Read image
                print(the_image.shape)
                # Convert to grayscale
                gray_scaled_img = data_prep_obj.convert_to_gray(the_image)
                # Perform image histogram equalization
                equalized_img = data_prep_obj.perform_hist_equalization(gray_scaled_img)
                # Resize the image
                resized_img = data_prep_obj.resize_image(img_h_size, img_w_size, equalized_img, 'bicubic')
                print(resized_img.shape)

                # Save processed image
                image_name = class_label + str(count) + '.jpg'
                process_img_path = os.path.abspath(final_processed_path) + '\\' + class_label + '\\' + image_name
                final_img_path = process_img_path.replace('\\', '/')
                data_prep_obj.save_final_image(process_img_path, resized_img)

                # all_images_list.append(os.path.abspath(subdir+'\\'+file))
                image_file_path = class_folder_path + '\\' + image_name   # use renamed file or could just use 'file'
                all_images_list.append(image_file_path)
               # print(os.path.abspath(subdir + '\\' + file))
                count += 1
            number_of_classes += 1
else:
    print("The image directory provided does not exist, please provide valid path")

print(number_of_classes)
# randomise and shuffle the list
shuffle(all_images_list)
# print(len(all_images_list))

# Prepare training and testing lists
print("Preparing training and testing lists")
print("--------------------------------------------------------------------")
train_images_list = all_images_list[int(len(all_images_list) * 0.00):int(len(all_images_list) * train_ratio)]
test_images_list = all_images_list[int(len(all_images_list) * (train_ratio + 0.01)):int(len(all_images_list) * 1.00)]
# print(len(train_images_list))
# X_train, X_test, y_train, y_test = train_test_split(all_images_list, all_images_list, test_size=0.20, random_state=42)

# Creating image class labels
train_images_labels = data_prep_obj.create_class_labels(train_images_list)
test_images_labels = data_prep_obj.create_class_labels(test_images_list)

# Convert the list to a unique list using set
class_names = set(class_names)

# Save image files
data_prep_obj.save_list('../train_list.txt', train_images_list)
data_prep_obj.save_list('../test_list.txt', test_images_list)
data_prep_obj.save_list('../train_labels.txt', train_images_labels)
data_prep_obj.save_list('../test_labels.txt', test_images_labels)
data_prep_obj.save_list('../class_names.txt', class_names)
print('--------------Image processing and data preparation completed successfully----------------------------------')
est_time = datetime.now() - startTime
print('It took about ' + est_time +' seconds' )
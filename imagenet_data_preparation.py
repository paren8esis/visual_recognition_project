# -*- coding: utf-8 -*-

# This script prepares the ImageNet data for the classifier.
# Meant to be run once.
#
# Dataset has been downloaded from: http://www.image-net.org/
#
# Folder structure:
#
#   datasets/
#       ImageNet/

import os

import utils


data_folder = os.path.join(os.getcwd(), 'datasets')
imagenet_folder = os.path.join(data_folder, 'ImageNet')

selected_classes = os.listdir(imagenet_folder)
selected_classes = [x.split('.')[0].split('-')[1] for x in selected_classes]

# Create a separate folder for each class
utils.create_folders([x + '_all' for x in selected_classes], imagenet_folder)

# Save the class labels in a .txt file
with open(os.path.join(imagenet_folder, 'selected_classes.txt'), 'a') as f:
    for label in selected_classes:
        f.write(label + '\n')

# Uncompress the data into the respective folders
utils.uncompress_folders(imagenet_folder, imagenet_folder)

# Select 100 random images from each class,
# and then store them in new folders.
number_of_images = 100
utils.select_random_images(selected_classes, imagenet_folder, imagenet_folder,
                           number_of_images)

# Compress each class folder into a separate .zip file for the classifier
utils.compress_folders(imagenet_folder, selected_classes)

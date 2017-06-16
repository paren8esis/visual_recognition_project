# -*- coding: utf-8 -*-

# Test code for the Watson Visual Recognition classifier.
# Before running this script, we assume that all data has been prepared.
#
# Folder structure:
#
#   custom_classifier.py
#   datasets/
#       grassias/
#       ImageNet/
#   *.zip files for training
#
# API: https://www.ibm.com/watson/developercloud/visual-recognition/api/v3/#classify_an_image

import os
from watson_developer_cloud import VisualRecognitionV3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.manifold import TSNE

import time

import utils

train_folder = os.path.join(os.getcwd(), 'datasets', 'ImageNet')
test_folder = os.path.join(os.getcwd(), 'datasets', 'grassias')

api_key = '4fa5700177e476ed183d077aa826ea1f6cc8a383'
version = '2016-05-20'
visual_recognition = VisualRecognitionV3(version, api_key=api_key)

# Print all classifiers that have been created
print("\nList of classifiers:")
classifiers_list = visual_recognition.list_classifiers()
print(classifiers_list)

delete_keep = '-'
if classifiers_list['classifiers'] != []:
    # Classifier already exists
    classifier_id = classifiers_list['classifiers'][0]['classifier_id']
    classifier_status = classifiers_list['classifiers'][0]['status']

    delete_keep = input('Delete or keep classifier? (d/k) ')
    while delete_keep not in ['d', 'D', 'k', 'K']:
        delete_keep = input('Invalid input. Delete or keep classifier? (d/k) ')

    if delete_keep in ['d', 'D']:
        visual_recognition.delete_classifier(classifier_id=classifier_id)

if delete_keep not in ['k', 'K']:
    # Train new classifier
    print('Creating the classifier...')

    # Training limitations:
    #   1) Maximum of 10,000 images or 100 MB per .zip file
    #   2) Minimum of 10 images per .zip file
    #   3) Maximum of 256 MB per training call
    #   4) Minimum recommended size of image is 32 x 32 pixels
    #
    # If these limitations are not met, an error '104: Connection reset by peer'
    # is raised.
    with open('airplane.zip', 'rb') as airplane, \
        open('car.zip', 'rb') as car, \
        open('bird.zip', 'rb') as bird, \
        open('cat.zip', 'rb') as cat, \
        open('dog.zip', 'rb') as dog:
            classifier = visual_recognition.create_classifier('imagenet5categories',
                                                              airplane_positive_examples=airplane,
                                                              car_positive_examples=car,
                                                              bird_positive_examples=bird,
                                                              cat_positive_examples=cat,
                                                              dog_positive_examples=dog)

    classifier_id = classifier['classifier_id']
    classifier_status = classifier['status']

    print("\nList of classifiers:")
    classifiers_list = visual_recognition.list_classifiers()
    print(classifiers_list)

    print('Classifier created! Now start training...')

# Check when the classifier is ready (ask every 5 mins)
while classifier_status == 'training':
    time.sleep(300)
    classifier = visual_recognition.get_classifier(classifier_id)
    classifier_status = classifier['status']

if classifier_status == 'ready':
    print('The classifier is ready!')
else:
    print('Something occured while training the classifier. Exiting...')
    print('Status: ', classifier_status)
    exit(1)

# Read the test images filenames
test_images = os.listdir(test_folder)
test_images_paths = [os.path.join(test_folder, x) for x in test_images]

# Read the classes
with open(os.path.join(train_folder, 'selected_classes.txt'), 'r') as f:
    selected_classes = f.read().rstrip('\n').split('\n')

# Classify test images and show results
prob_vectors = []
i = 1
fig_i = 1
fig = plt.figure(fig_i)

for x in range(len(test_images_paths)):
    print('\nClassifying image ' + test_images[x])
    res = visual_recognition.classify(open(test_images_paths[x], 'rb'),
                                      classifier_ids=classifier_id,
                                      threshold=0.0)
    print('Classified image ' + test_images[x])

    res = res['images'][0]['classifiers'][0]['classes']
    prob_vector = np.asarray([x['score'] for x in res])
    prob_vectors.append(prob_vector)
    res_sorted = sorted(res, key=operator.itemgetter('score'), reverse=True)

    if ((i-1) % 8 == 0) and (i != 1):
        fig_i += 1
        fig = plt.figure(fig_i)
        i = 1
    test_image_arr = cv2.imread(test_images_paths[x])

    plt.subplot(240 + i)
    plt.imshow(test_image_arr[:, :, ::-1])
    plt.axis('off')
    plt.subplot(240 + i + 1)
    plt.bar(range(len(selected_classes)), [x['score'] for x in res_sorted])
    plt.xticks(range(len(selected_classes)), [x['class'] for x in res_sorted],
               rotation=65)
#    plt.text(0, 0, ''.join([x['class'] + ': ' + str(x['score']) + '\n' for x in res_sorted]))
    plt.tight_layout()

    i += 2
prob_vectors = np.asarray(prob_vectors)

plt.show()

# Run t-SNE on the test set,
# and visualize the results in a 2D plot.
model = TSNE(n_components=2, random_state=0)
x = model.fit_transform(prob_vectors)

utils.plot_embedding(x, test_images_paths, len(test_images))

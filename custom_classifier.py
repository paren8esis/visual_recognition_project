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
#       test_set/
#   *.zip files for training
#
# API: https://www.ibm.com/watson/developercloud/visual-recognition/api/v3/#classify_an_image

import os
from watson_developer_cloud import VisualRecognitionV3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from sklearn import metrics

import time

import utils

train_folder = os.path.join(os.getcwd(), 'datasets', 'ImageNet')
test_folder = os.path.join(os.getcwd(), 'datasets', 'test_set')

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
#prob_vectors = []
#i = 1
#fig_i = 1
#fig = plt.figure(fig_i)
#
#for x in range(len(test_images_paths)):
#    print('\nClassifying image ' + test_images[x])
#    res = visual_recognition.classify(open(test_images_paths[x], 'rb'),
#                                      classifier_ids=classifier_id,
#                                      threshold=0.0)
#    print('Classified image ' + test_images[x])
#
#    res = res['images'][0]['classifiers'][0]['classes']
#    prob_vector = np.asarray([x['score'] for x in res])
#    prob_vectors.append(prob_vector)
#    res_sorted = sorted(res, key=operator.itemgetter('score'), reverse=True)
#
#    if ((i-1) % 8 == 0) and (i != 1):
#        fig_i += 1
#        fig = plt.figure(fig_i)
#        i = 1
#    test_image_arr = cv2.imread(test_images_paths[x])
#
#    plt.subplot(240 + i)
#    plt.imshow(test_image_arr[:, :, ::-1])
#    plt.axis('off')
#    plt.subplot(240 + i + 1)
#    plt.bar(range(len(selected_classes)), [x['score'] for x in res_sorted])
#    plt.xticks(range(len(selected_classes)), [x['class'] for x in res_sorted],
#               rotation=65)
##    plt.text(0, 0, ''.join([x['class'] + ': ' + str(x['score']) + '\n' for x in res_sorted]))
#    plt.tight_layout()
#
#    i += 2
#prob_vectors = np.asarray(prob_vectors)
#
#plt.show()

#np.save('prob_vectors', prob_vectors)
prob_vectors = np.load('prob_vectors.npy')

# Run PCA on the test set
# and visualize the results in a 2D plot.
model_PCA = PCA(n_components=2)
x_PCA = model_PCA.fit_transform(prob_vectors)

utils.plot_embedding(x_PCA, test_images_paths, len(test_images))

# Run Kernel PCA on the test set
# and visualize the results in a 2D plot.
model_KernelPCA_cosine = KernelPCA(n_components=2, kernel='cosine')
x_KernelPCA_cosine = model_KernelPCA_cosine.fit_transform(prob_vectors)

utils.plot_embedding(x_KernelPCA_cosine, test_images_paths, len(test_images))

model_KernelPCA_rbf = KernelPCA(n_components=2, kernel='rbf')
x_KernelPCA_rbf = model_KernelPCA_rbf.fit_transform(prob_vectors)

utils.plot_embedding(x_KernelPCA_rbf, test_images_paths, len(test_images))

# Run t-SNE on the test set,
# and visualize the results in a 2D plot.
model_tSNE_lr100 = TSNE(n_components=2, random_state=0, learning_rate=100)
x_tSNE_lr100 = model_tSNE_lr100.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr100, test_images_paths, len(test_images))

model_tSNE_lr100_mExact = TSNE(n_components=2, random_state=0,
                               learning_rate=100, method='exact')
x_tSNE_lr100_mExact = model_tSNE_lr100_mExact.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr100_mExact, test_images_paths, len(test_images))

model_tSNE_lr200_mExact = TSNE(n_components=2, random_state=0,
                               learning_rate=200, method='exact')
x_tSNE_lr200_mExact = model_tSNE_lr200_mExact.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr200_mExact, test_images_paths, len(test_images))

# Run Isomap on the test set,
# and visualize the results in a 2D plot.
model_isomap = Isomap(n_components=2)
x_isomap = model_isomap.fit_transform(prob_vectors)

utils.plot_embedding(x_isomap, test_images_paths, len(test_images))

# Clustering

# Get the true labels for each image
true_labels = utils.get_labels_imagenet(test_images)
labels_replace = {key: value for (key, value) in [(x, y) for x, y in zip(selected_classes, range(len(selected_classes)))]}
true_labels_int = [labels_replace[x] for x in true_labels]

# Run k-Means clustering on the image embeddings.
model_kmeans_PCA = KMeans(n_clusters=len(selected_classes))
x_kmeans_PCA = model_kmeans_PCA.fit_predict(x_PCA)

fig = plt.figure()
plt.scatter(x_PCA[:, 0], x_PCA[:, 1], c=x_kmeans_PCA)
plt.title('PCA + kmeans')
plt.show()

model_kmeans_KernelPCA_cosine = KMeans(n_clusters=len(selected_classes))
x_kmeans_KernelPCA_cosine = model_kmeans_KernelPCA_cosine.fit_predict(x_KernelPCA_cosine)

fig = plt.figure()
plt.scatter(x_KernelPCA_cosine[:, 0], x_PCA[:, 1], c=x_kmeans_KernelPCA_cosine)
plt.title('Kernel PCA (cosine) + kmeans')
plt.show()

model_kmeans_KernelPCA_rbf = KMeans(n_clusters=len(selected_classes))
x_kmeans_KernelPCA_rbf = model_kmeans_KernelPCA_rbf.fit_predict(x_KernelPCA_rbf)

fig = plt.figure()
plt.scatter(x_KernelPCA_rbf[:, 0], x_PCA[:, 1], c=x_kmeans_KernelPCA_rbf)
plt.title('Kernel PCA (rbf) + kmeans')
plt.show()

model_kmeans_tSNE_lr100 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr100 = model_kmeans_tSNE_lr100.fit_predict(x_tSNE_lr100)

fig = plt.figure()
plt.scatter(x_tSNE_lr100[:, 0], x_PCA[:, 1], c=x_kmeans_tSNE_lr100)
plt.title('t-SNE (learning rate = 100) + kmeans')
plt.show()

model_kmeans_tSNE_lr100_mExact = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr100_mExact = model_kmeans_tSNE_lr100_mExact.fit_predict(x_tSNE_lr100_mExact)

fig = plt.figure()
plt.scatter(x_tSNE_lr100_mExact[:, 0], x_PCA[:, 1], c=x_kmeans_tSNE_lr100_mExact)
plt.title('t-SNE (learning rate = 100, method = exact) + kmeans')
plt.show()

model_kmeans_tSNE_lr200_mExact = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr200_mExact = model_kmeans_tSNE_lr100_mExact.fit_predict(x_tSNE_lr200_mExact)

fig = plt.figure()
plt.scatter(x_tSNE_lr200_mExact[:, 0], x_PCA[:, 1], c=x_kmeans_tSNE_lr200_mExact)
plt.title('t-SNE (learning rate = 200, method = exact) + kmeans')
plt.show()

model_kmeans_isomap = KMeans(n_clusters=len(selected_classes))
x_kmeans_isomap = model_kmeans_isomap.fit_predict(x_isomap)

fig = plt.figure()
plt.scatter(x_isomap[:, 0], x_PCA[:, 1], c=x_kmeans_isomap)
plt.title('Isomap + kmeans')
plt.show()

# Cluster evaluation

# Adjusted Rand Score
ars_kmeans_PCA = metrics.adjusted_rand_score(true_labels_int, x_kmeans_PCA)
ars_kmeans_KernelPCA_cosine = metrics.adjusted_rand_score(true_labels_int, x_kmeans_KernelPCA_cosine)
ars_kmeans_KernelPCA_rbf = metrics.adjusted_rand_score(true_labels_int, x_kmeans_KernelPCA_rbf)
ars_kmeans_tSNE_lr100 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr100)
ars_kmeans_tSNE_lr100_mExact = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr100_mExact)
ars_kmeans_tSNE_lr200_mExact = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
ars_kmeans_isomap = metrics.adjusted_rand_score(true_labels_int, x_kmeans_isomap)

ars = [ars_kmeans_PCA,
       ars_kmeans_KernelPCA_cosine,
       ars_kmeans_KernelPCA_rbf,
       ars_kmeans_tSNE_lr100,
       ars_kmeans_tSNE_lr100_mExact,
       ars_kmeans_tSNE_lr200_mExact,
       ars_kmeans_isomap]

# Mutual Info Score
mis_kmeans_PCA = metrics.mutual_info_score(true_labels_int, x_kmeans_PCA)
mis_kmeans_KernelPCA_cosine = metrics.mutual_info_score(true_labels_int, x_kmeans_KernelPCA_cosine)
mis_kmeans_KernelPCA_rbf = metrics.mutual_info_score(true_labels_int, x_kmeans_KernelPCA_rbf)
mis_kmeans_tSNE_lr100 = metrics.mutual_info_score(true_labels_int, x_kmeans_tSNE_lr100)
mis_kmeans_tSNE_lr100_mExact = metrics.mutual_info_score(true_labels_int, x_kmeans_tSNE_lr100_mExact)
mis_kmeans_tSNE_lr200_mExact = metrics.mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
mis_kmeans_isomap = metrics.mutual_info_score(true_labels_int, x_kmeans_isomap)

mis = [mis_kmeans_PCA,
       mis_kmeans_KernelPCA_cosine,
       mis_kmeans_KernelPCA_rbf,
       mis_kmeans_tSNE_lr100,
       mis_kmeans_tSNE_lr100_mExact,
       mis_kmeans_tSNE_lr200_mExact,
       mis_kmeans_isomap]

# Adjusted Mutual Info Score
amis_kmeans_PCA = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_PCA)
amis_kmeans_KernelPCA_cosine = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_KernelPCA_cosine)
amis_kmeans_KernelPCA_rbf = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_KernelPCA_rbf)
amis_kmeans_tSNE_lr100 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr100)
amis_kmeans_tSNE_lr100_mExact = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr100_mExact)
amis_kmeans_tSNE_lr200_mExact = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
amis_kmeans_isomap = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_isomap)

amis = [amis_kmeans_PCA,
       amis_kmeans_KernelPCA_cosine,
       amis_kmeans_KernelPCA_rbf,
       amis_kmeans_tSNE_lr100,
       amis_kmeans_tSNE_lr100_mExact,
       amis_kmeans_tSNE_lr200_mExact,
       amis_kmeans_isomap]

# Homogeneity Score
hs_kmeans_PCA = metrics.homogeneity_score(true_labels_int, x_kmeans_PCA)
hs_kmeans_KernelPCA_cosine = metrics.homogeneity_score(true_labels_int, x_kmeans_KernelPCA_cosine)
hs_kmeans_KernelPCA_rbf = metrics.homogeneity_score(true_labels_int, x_kmeans_KernelPCA_rbf)
hs_kmeans_tSNE_lr100 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr100)
hs_kmeans_tSNE_lr100_mExact = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr100_mExact)
hs_kmeans_tSNE_lr200_mExact = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
hs_kmeans_isomap = metrics.homogeneity_score(true_labels_int, x_kmeans_isomap)

hs = [hs_kmeans_PCA,
      hs_kmeans_KernelPCA_cosine,
      hs_kmeans_KernelPCA_rbf,
      hs_kmeans_tSNE_lr100,
      hs_kmeans_tSNE_lr100_mExact,
      hs_kmeans_tSNE_lr200_mExact,
      hs_kmeans_isomap]

# Completeness Score
cs_kmeans_PCA = metrics.completeness_score(true_labels_int, x_kmeans_PCA)
cs_kmeans_KernelPCA_cosine = metrics.completeness_score(true_labels_int, x_kmeans_KernelPCA_cosine)
cs_kmeans_KernelPCA_rbf = metrics.completeness_score(true_labels_int, x_kmeans_KernelPCA_rbf)
cs_kmeans_tSNE_lr100 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr100)
cs_kmeans_tSNE_lr100_mExact = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr100_mExact)
cs_kmeans_tSNE_lr200_mExact = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
cs_kmeans_isomap = metrics.completeness_score(true_labels_int, x_kmeans_isomap)

cs = [cs_kmeans_PCA,
      cs_kmeans_KernelPCA_cosine,
      cs_kmeans_KernelPCA_rbf,
      cs_kmeans_tSNE_lr100,
      cs_kmeans_tSNE_lr100_mExact,
      cs_kmeans_tSNE_lr200_mExact,
      cs_kmeans_isomap]

# V-measure Score
vms_kmeans_PCA = metrics.v_measure_score(true_labels_int, x_kmeans_PCA)
vms_kmeans_KernelPCA_cosine = metrics.v_measure_score(true_labels_int, x_kmeans_KernelPCA_cosine)
vms_kmeans_KernelPCA_rbf = metrics.v_measure_score(true_labels_int, x_kmeans_KernelPCA_rbf)
vms_kmeans_tSNE_lr100 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr100)
vms_kmeans_tSNE_lr100_mExact = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr100_mExact)
vms_kmeans_tSNE_lr200_mExact = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
vms_kmeans_isomap = metrics.v_measure_score(true_labels_int, x_kmeans_isomap)

vms = [vms_kmeans_PCA,
      vms_kmeans_KernelPCA_cosine,
      vms_kmeans_KernelPCA_rbf,
      vms_kmeans_tSNE_lr100,
      vms_kmeans_tSNE_lr100_mExact,
      vms_kmeans_tSNE_lr200_mExact,
      vms_kmeans_isomap]

# Fowlkes-Mallows Score
fms_kmeans_PCA = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_PCA)
fms_kmeans_KernelPCA_cosine = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_KernelPCA_cosine)
fms_kmeans_KernelPCA_rbf = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_KernelPCA_rbf)
fms_kmeans_tSNE_lr100 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr100)
fms_kmeans_tSNE_lr100_mExact = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr100_mExact)
fms_kmeans_tSNE_lr200_mExact = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
fms_kmeans_isomap = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_isomap)

fms = [fms_kmeans_PCA,
      fms_kmeans_KernelPCA_cosine,
      fms_kmeans_KernelPCA_rbf,
      fms_kmeans_tSNE_lr100,
      fms_kmeans_tSNE_lr100_mExact,
      fms_kmeans_tSNE_lr200_mExact,
      fms_kmeans_isomap]


# Plot the accuracy results in a bar plot
score_names = ['ars', 'mis', 'amis', 'hs', 'cs', 'vms', 'fms']
scores = [ars, mis, amis, hs, cs, vms, fms]
clustering_methods = ['PCA', 'KernelPCA (cosine)', 'KernelPCA (rbf)',
                    't-SNE\n(lr=100)', 't-SNE\n(lr=100,\nmethod=exact)',
                    't-SNE\n(lr=200,\nmethod=exact)', 'isomap']
utils.plot_cluster_evaluation_scores(scores, score_names, clustering_methods)

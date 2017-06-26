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

# Get the true labels for each image
true_labels = utils.get_labels_imagenet(test_images)
labels_replace = {key: value for (key, value) in [(x, y) for x, y in zip(selected_classes, range(len(selected_classes)))]}
true_labels_int = [labels_replace[x] for x in true_labels]

# Perform clustering with k-Means on the probability vectors
model_init = KMeans(n_clusters=len(selected_classes))
x_init = model_init.fit_predict(prob_vectors)

# Evaluate the results
ars_init = metrics.adjusted_rand_score(true_labels_int, x_init)
mis_init = metrics.mutual_info_score(true_labels_int, x_init)
amis_init = metrics.adjusted_mutual_info_score(true_labels_int, x_init)
hs_init = metrics.homogeneity_score(true_labels_int, x_init)
cs_init = metrics.completeness_score(true_labels_int, x_init)
vms_init = metrics.v_measure_score(true_labels_int, x_init)
fms_init = metrics.fowlkes_mallows_score(true_labels_int, x_init)


## Run PCA on the test set
## and visualize the results in a 2D plot.
#model_PCA = PCA(n_components=2)
#x_PCA = model_PCA.fit_transform(prob_vectors)
#
#utils.plot_embedding(x_PCA, test_images_paths, len(test_images))
#
## Run Kernel PCA on the test set
## and visualize the results in a 2D plot.
#model_KernelPCA_cosine = KernelPCA(n_components=2, kernel='cosine')
#x_KernelPCA_cosine = model_KernelPCA_cosine.fit_transform(prob_vectors)
#
#utils.plot_embedding(x_KernelPCA_cosine, test_images_paths, len(test_images))
#
#model_KernelPCA_rbf = KernelPCA(n_components=2, kernel='rbf')
#x_KernelPCA_rbf = model_KernelPCA_rbf.fit_transform(prob_vectors)
#
#utils.plot_embedding(x_KernelPCA_rbf, test_images_paths, len(test_images))
#
## Run Isomap on the test set,
## and visualize the results in a 2D plot.
#model_isomap = Isomap(n_components=2)
#x_isomap = model_isomap.fit_transform(prob_vectors)
#
#utils.plot_embedding(x_isomap, test_images_paths, len(test_images))


# Run t-SNE on the test set,
# and visualize the results in a 2D plot.
model_tSNE_lr100 = TSNE(n_components=2, random_state=0, learning_rate=100)
x_tSNE_lr100 = model_tSNE_lr100.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr100, test_images_paths, len(test_images))

#
model_tSNE_lr100_per5 = TSNE(n_components=2, random_state=0, learning_rate=100,
                             perplexity=5)
x_tSNE_lr100_per5 = model_tSNE_lr100_per5.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr100_per5, test_images_paths, len(test_images))

#
model_tSNE_lr100_per10 = TSNE(n_components=2, random_state=0, learning_rate=100,
                              perplexity=10)
x_tSNE_lr100_per10 = model_tSNE_lr100_per10.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr100_per10, test_images_paths, len(test_images))

#
model_tSNE_lr100_per15 = TSNE(n_components=2, random_state=0, learning_rate=100,
                              perplexity=15)
x_tSNE_lr100_per15 = model_tSNE_lr100_per15.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr100_per15, test_images_paths, len(test_images))

#
model_tSNE_lr100_per20 = TSNE(n_components=2, random_state=0, learning_rate=100,
                              perplexity=20)
x_tSNE_lr100_per20 = model_tSNE_lr100_per20.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr100_per20, test_images_paths, len(test_images))


#
model_tSNE_lr200 = TSNE(n_components=2, random_state=0,
                        learning_rate=200)
x_tSNE_lr200 = model_tSNE_lr200.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr200, test_images_paths, len(test_images))

#
model_tSNE_lr200_mExact = TSNE(n_components=2, random_state=0,
                               learning_rate=200, method='exact')
x_tSNE_lr200_mExact = model_tSNE_lr200_mExact.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr200_mExact, test_images_paths, len(test_images))

#
model_tSNE_lr200_mExact_per5 = TSNE(n_components=2, random_state=0,
                               learning_rate=200, method='exact',
                               perplexity=5)
x_tSNE_lr200_mExact_per5 = model_tSNE_lr200_mExact_per5.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr200_mExact_per5, test_images_paths, len(test_images))

#
model_tSNE_lr200_mExact_per10 = TSNE(n_components=2, random_state=0,
                               learning_rate=200, method='exact',
                               perplexity=10)
x_tSNE_lr200_mExact_per10 = model_tSNE_lr200_mExact_per10.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr200_mExact_per10, test_images_paths, len(test_images))

#
model_tSNE_lr200_mExact_per15 = TSNE(n_components=2, random_state=0,
                               learning_rate=200, method='exact',
                               perplexity=15)
x_tSNE_lr200_mExact_per15 = model_tSNE_lr200_mExact_per15.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr200_mExact_per15, test_images_paths, len(test_images))

#
model_tSNE_lr200_mExact_per20 = TSNE(n_components=2, random_state=0,
                               learning_rate=200, method='exact',
                               perplexity=20)
x_tSNE_lr200_mExact_per20 = model_tSNE_lr200_mExact_per20.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr200_mExact_per20, test_images_paths, len(test_images))

#
model_tSNE_lr200_mExact_per40 = TSNE(n_components=2, random_state=0,
                               learning_rate=200, method='exact',
                               perplexity=40)
x_tSNE_lr200_mExact_per40 = model_tSNE_lr200_mExact_per40.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr200_mExact_per40, test_images_paths, len(test_images))

#
model_tSNE_lr200_mExact_per50 = TSNE(n_components=2, random_state=0,
                               learning_rate=200, method='exact',
                               perplexity=50)
x_tSNE_lr200_mExact_per50 = model_tSNE_lr200_mExact_per50.fit_transform(prob_vectors)

utils.plot_embedding(x_tSNE_lr200_mExact_per50, test_images_paths, len(test_images))


# Clustering

# Run k-Means clustering on the image embeddings.
#model_kmeans_PCA = KMeans(n_clusters=len(selected_classes))
#x_kmeans_PCA = model_kmeans_PCA.fit_predict(x_PCA)
#
#fig = plt.figure()
#plt.scatter(x_PCA[:, 0], x_PCA[:, 1], c=x_kmeans_PCA)
#plt.title('PCA + kmeans')
#plt.show()
#
#model_kmeans_KernelPCA_cosine = KMeans(n_clusters=len(selected_classes))
#x_kmeans_KernelPCA_cosine = model_kmeans_KernelPCA_cosine.fit_predict(x_KernelPCA_cosine)
#
#fig = plt.figure()
#plt.scatter(x_KernelPCA_cosine[:, 0], x_KernelPCA_cosine[:, 1], c=x_kmeans_KernelPCA_cosine)
#plt.title('Kernel PCA (cosine) + kmeans')
#plt.show()
#
#model_kmeans_KernelPCA_rbf = KMeans(n_clusters=len(selected_classes))
#x_kmeans_KernelPCA_rbf = model_kmeans_KernelPCA_rbf.fit_predict(x_KernelPCA_rbf)
#
#fig = plt.figure()
#plt.scatter(x_KernelPCA_rbf[:, 0], x_KernelPCA_rbf[:, 1], c=x_kmeans_KernelPCA_rbf)
#plt.title('Kernel PCA (rbf) + kmeans')
#plt.show()
#
#model_kmeans_isomap = KMeans(n_clusters=len(selected_classes))
#x_kmeans_isomap = model_kmeans_isomap.fit_predict(x_isomap)
#
#fig = plt.figure()
#plt.scatter(x_isomap[:, 0], x_isomap[:, 1], c=x_kmeans_isomap)
#plt.title('Isomap + kmeans')
#plt.show()

model_kmeans_tSNE_lr100 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr100 = model_kmeans_tSNE_lr100.fit_predict(x_tSNE_lr100)

fig = plt.figure()
plt.scatter(x_tSNE_lr100[:, 0], x_tSNE_lr100[:, 1], c=x_kmeans_tSNE_lr100)
plt.title('t-SNE (learning rate = 100) + kmeans')
plt.show()

model_kmeans_tSNE_lr100_per5 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr100_per5 = model_kmeans_tSNE_lr100_per5.fit_predict(x_tSNE_lr100_per5)

fig = plt.figure()
plt.scatter(x_tSNE_lr100_per5[:, 0], x_tSNE_lr100_per5[:, 1], c=x_kmeans_tSNE_lr100_per5)
plt.title('t-SNE (learning rate = 100, perplexity = 5) + kmeans')
plt.show()

model_kmeans_tSNE_lr100_per10 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr100_per10 = model_kmeans_tSNE_lr100_per10.fit_predict(x_tSNE_lr100_per10)

fig = plt.figure()
plt.scatter(x_tSNE_lr100_per10[:, 0], x_tSNE_lr100_per10[:, 1], c=x_kmeans_tSNE_lr100_per10)
plt.title('t-SNE (learning rate = 100, perplexity = 10) + kmeans')
plt.show()

model_kmeans_tSNE_lr100_per15 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr100_per15 = model_kmeans_tSNE_lr100_per15.fit_predict(x_tSNE_lr100_per15)

fig = plt.figure()
plt.scatter(x_tSNE_lr100_per15[:, 0], x_tSNE_lr100_per15[:, 1], c=x_kmeans_tSNE_lr100_per15)
plt.title('t-SNE (learning rate = 100, perplexity = 15) + kmeans')
plt.show()

model_kmeans_tSNE_lr100_per20 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr100_per20 = model_kmeans_tSNE_lr100_per20.fit_predict(x_tSNE_lr100_per20)

fig = plt.figure()
plt.scatter(x_tSNE_lr100_per20[:, 0], x_tSNE_lr100_per20[:, 1], c=x_kmeans_tSNE_lr100_per20)
plt.title('t-SNE (learning rate = 100, perplexity = 20) + kmeans')
plt.show()


model_kmeans_tSNE_lr200 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr200 = model_kmeans_tSNE_lr200.fit_predict(x_tSNE_lr200)

fig = plt.figure()
plt.scatter(x_tSNE_lr200[:, 0], x_tSNE_lr200[:, 1], c=x_kmeans_tSNE_lr200)
plt.title('t-SNE (learning rate = 200) + kmeans')
plt.show()

model_kmeans_tSNE_lr200_mExact = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr200_mExact = model_kmeans_tSNE_lr200_mExact.fit_predict(x_tSNE_lr200_mExact)

fig = plt.figure()
plt.scatter(x_tSNE_lr200_mExact[:, 0], x_tSNE_lr200_mExact[:, 1], c=x_kmeans_tSNE_lr200_mExact)
plt.title('t-SNE (learning rate = 200, method = exact) + kmeans')
plt.show()

model_kmeans_tSNE_lr200_mExact_per5 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr200_mExact_per5 = model_kmeans_tSNE_lr200_mExact_per5.fit_predict(x_tSNE_lr200_mExact_per5)

fig = plt.figure()
plt.scatter(x_tSNE_lr200_mExact_per5[:, 0], x_tSNE_lr200_mExact_per5[:, 1], c=x_kmeans_tSNE_lr200_mExact_per5)
plt.title('t-SNE (learning rate = 200, method = exact, perplexity = 5) + kmeans')
plt.show()

model_kmeans_tSNE_lr200_mExact_per10 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr200_mExact_per10 = model_kmeans_tSNE_lr200_mExact_per10.fit_predict(x_tSNE_lr200_mExact_per10)

fig = plt.figure()
plt.scatter(x_tSNE_lr200_mExact_per10[:, 0], x_tSNE_lr200_mExact_per10[:, 1], c=x_kmeans_tSNE_lr200_mExact_per10)
plt.title('t-SNE (learning rate = 200, method = exact, perplexity = 10) + kmeans')
plt.show()

model_kmeans_tSNE_lr200_mExact_per15 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr200_mExact_per15 = model_kmeans_tSNE_lr200_mExact_per15.fit_predict(x_tSNE_lr200_mExact_per15)

fig = plt.figure()
plt.scatter(x_tSNE_lr200_mExact_per15[:, 0], x_tSNE_lr200_mExact_per15[:, 1], c=x_kmeans_tSNE_lr200_mExact_per15)
plt.title('t-SNE (learning rate = 200, method = exact, perplexity = 15) + kmeans')
plt.show()

model_kmeans_tSNE_lr200_mExact_per20 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr200_mExact_per20 = model_kmeans_tSNE_lr200_mExact_per20.fit_predict(x_tSNE_lr200_mExact_per20)

fig = plt.figure()
plt.scatter(x_tSNE_lr200_mExact_per20[:, 0], x_tSNE_lr200_mExact_per20[:, 1], c=x_kmeans_tSNE_lr200_mExact_per20)
plt.title('t-SNE (learning rate = 200, method = exact, perplexity = 20) + kmeans')
plt.show()

model_kmeans_tSNE_lr200_mExact_per40 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr200_mExact_per40 = model_kmeans_tSNE_lr200_mExact_per40.fit_predict(x_tSNE_lr200_mExact_per40)

fig = plt.figure()
plt.scatter(x_tSNE_lr200_mExact_per40[:, 0], x_tSNE_lr200_mExact_per40[:, 1], c=x_kmeans_tSNE_lr200_mExact_per40)
plt.title('t-SNE (learning rate = 200, method = exact, perplexity = 40) + kmeans')
plt.show()

model_kmeans_tSNE_lr200_mExact_per50 = KMeans(n_clusters=len(selected_classes))
x_kmeans_tSNE_lr200_mExact_per50 = model_kmeans_tSNE_lr200_mExact_per50.fit_predict(x_tSNE_lr200_mExact_per50)

fig = plt.figure()
plt.scatter(x_tSNE_lr200_mExact_per50[:, 0], x_tSNE_lr200_mExact_per50[:, 1], c=x_kmeans_tSNE_lr200_mExact_per50)
plt.title('t-SNE (learning rate = 200, method = exact, perplexity = 50) + kmeans')
plt.show()


# Cluster evaluation

# Adjusted Rand Score
#ars_kmeans_PCA = metrics.adjusted_rand_score(true_labels_int, x_kmeans_PCA)
#ars_kmeans_KernelPCA_cosine = metrics.adjusted_rand_score(true_labels_int, x_kmeans_KernelPCA_cosine)
#ars_kmeans_KernelPCA_rbf = metrics.adjusted_rand_score(true_labels_int, x_kmeans_KernelPCA_rbf)
#ars_kmeans_isomap = metrics.adjusted_rand_score(true_labels_int, x_kmeans_isomap)
ars_kmeans_tSNE_lr100 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr100)
ars_kmeans_tSNE_lr100_per10 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr100_per10)
ars_kmeans_tSNE_lr100_per15 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr100_per15)
ars_kmeans_tSNE_lr100_per20 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr100_per20)
ars_kmeans_tSNE_lr200 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr200)
ars_kmeans_tSNE_lr200_mExact = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
ars_kmeans_tSNE_lr200_mExact_per10 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per10)
ars_kmeans_tSNE_lr200_mExact_per15 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per15)
ars_kmeans_tSNE_lr200_mExact_per20 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per20)
ars_kmeans_tSNE_lr200_mExact_per40 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per40)
ars_kmeans_tSNE_lr200_mExact_per50 = metrics.adjusted_rand_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per50)

#ars = [ars_init,
#       ars_kmeans_PCA,
#       ars_kmeans_KernelPCA_cosine,
#       ars_kmeans_KernelPCA_rbf,
#       ars_kmeans_tSNE_lr100,
#       ars_kmeans_tSNE_lr200,
#       ars_kmeans_tSNE_lr200_mExact,
#       ars_kmeans_isomap]

ars = [ars_init,
       ars_kmeans_tSNE_lr100,
       ars_kmeans_tSNE_lr100_per10,
       ars_kmeans_tSNE_lr100_per15,
       ars_kmeans_tSNE_lr100_per20,
       ars_kmeans_tSNE_lr200,
       ars_kmeans_tSNE_lr200_mExact,
       ars_kmeans_tSNE_lr200_mExact_per10,
       ars_kmeans_tSNE_lr200_mExact_per15,
       ars_kmeans_tSNE_lr200_mExact_per20,
       ars_kmeans_tSNE_lr200_mExact_per40,
       ars_kmeans_tSNE_lr200_mExact_per50
       ]

# Mutual Info Score
#mis_kmeans_PCA = metrics.mutual_info_score(true_labels_int, x_kmeans_PCA)
#mis_kmeans_KernelPCA_cosine = metrics.mutual_info_score(true_labels_int, x_kmeans_KernelPCA_cosine)
#mis_kmeans_KernelPCA_rbf = metrics.mutual_info_score(true_labels_int, x_kmeans_KernelPCA_rbf)
#mis_kmeans_isomap = metrics.mutual_info_score(true_labels_int, x_kmeans_isomap)
#mis_kmeans_tSNE_lr100 = metrics.mutual_info_score(true_labels_int, x_kmeans_tSNE_lr100)
#mis_kmeans_tSNE_lr200 = metrics.mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200)
#mis_kmeans_tSNE_lr200_mExact = metrics.mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
#
#mis = [mis_init,
#       mis_kmeans_PCA,
#       mis_kmeans_KernelPCA_cosine,
#       mis_kmeans_KernelPCA_rbf,
#       mis_kmeans_tSNE_lr100,
#       mis_kmeans_tSNE_lr200,
#       mis_kmeans_tSNE_lr200_mExact,
#       mis_kmeans_isomap]

# Adjusted Mutual Info Score
#amis_kmeans_PCA = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_PCA)
#amis_kmeans_KernelPCA_cosine = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_KernelPCA_cosine)
#amis_kmeans_KernelPCA_rbf = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_KernelPCA_rbf)
#amis_kmeans_isomap = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_isomap)
amis_kmeans_tSNE_lr100 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr100)
amis_kmeans_tSNE_lr100_per10 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr100_per10)
amis_kmeans_tSNE_lr100_per15 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr100_per15)
amis_kmeans_tSNE_lr100_per20 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr100_per20)
amis_kmeans_tSNE_lr200 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200)
amis_kmeans_tSNE_lr200_mExact = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
amis_kmeans_tSNE_lr200_mExact_per10 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per10)
amis_kmeans_tSNE_lr200_mExact_per15 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per15)
amis_kmeans_tSNE_lr200_mExact_per20 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per20)
amis_kmeans_tSNE_lr200_mExact_per40 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per40)
amis_kmeans_tSNE_lr200_mExact_per50 = metrics.adjusted_mutual_info_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per50)

amis = [amis_init,
        amis_kmeans_tSNE_lr100,
        amis_kmeans_tSNE_lr100_per10,
        amis_kmeans_tSNE_lr100_per15,
        amis_kmeans_tSNE_lr100_per20,
        amis_kmeans_tSNE_lr200,
        amis_kmeans_tSNE_lr200_mExact,
        amis_kmeans_tSNE_lr200_mExact_per10,
        amis_kmeans_tSNE_lr200_mExact_per15,
        amis_kmeans_tSNE_lr200_mExact_per20,
        amis_kmeans_tSNE_lr200_mExact_per40,
        amis_kmeans_tSNE_lr200_mExact_per50
        ]

# Homogeneity Score
#hs_kmeans_PCA = metrics.homogeneity_score(true_labels_int, x_kmeans_PCA)
#hs_kmeans_KernelPCA_cosine = metrics.homogeneity_score(true_labels_int, x_kmeans_KernelPCA_cosine)
#hs_kmeans_KernelPCA_rbf = metrics.homogeneity_score(true_labels_int, x_kmeans_KernelPCA_rbf)
#hs_kmeans_isomap = metrics.homogeneity_score(true_labels_int, x_kmeans_isomap)
hs_kmeans_tSNE_lr100 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr100)
hs_kmeans_tSNE_lr100_per10 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr100_per10)
hs_kmeans_tSNE_lr100_per15 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr100_per15)
hs_kmeans_tSNE_lr100_per20 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr100_per20)
hs_kmeans_tSNE_lr200 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr200)
hs_kmeans_tSNE_lr200_mExact = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
hs_kmeans_tSNE_lr200_mExact_per10 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per10)
hs_kmeans_tSNE_lr200_mExact_per15 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per15)
hs_kmeans_tSNE_lr200_mExact_per20 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per20)
hs_kmeans_tSNE_lr200_mExact_per40 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per40)
hs_kmeans_tSNE_lr200_mExact_per50 = metrics.homogeneity_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per50)

hs = [hs_init,
      hs_kmeans_tSNE_lr100,
      hs_kmeans_tSNE_lr100_per10,
      hs_kmeans_tSNE_lr100_per15,
      hs_kmeans_tSNE_lr100_per20,
      hs_kmeans_tSNE_lr200,
      hs_kmeans_tSNE_lr200_mExact,
      hs_kmeans_tSNE_lr200_mExact_per10,
      hs_kmeans_tSNE_lr200_mExact_per15,
      hs_kmeans_tSNE_lr200_mExact_per20,
      hs_kmeans_tSNE_lr200_mExact_per40,
      hs_kmeans_tSNE_lr200_mExact_per50
      ]

# Completeness Score
#cs_kmeans_PCA = metrics.completeness_score(true_labels_int, x_kmeans_PCA)
#cs_kmeans_KernelPCA_cosine = metrics.completeness_score(true_labels_int, x_kmeans_KernelPCA_cosine)
#cs_kmeans_KernelPCA_rbf = metrics.completeness_score(true_labels_int, x_kmeans_KernelPCA_rbf)
#cs_kmeans_isomap = metrics.completeness_score(true_labels_int, x_kmeans_isomap)
cs_kmeans_tSNE_lr100 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr100)
cs_kmeans_tSNE_lr100_per10 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr100_per10)
cs_kmeans_tSNE_lr100_per15 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr100_per15)
cs_kmeans_tSNE_lr100_per20 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr100_per20)
cs_kmeans_tSNE_lr200 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr200)
cs_kmeans_tSNE_lr200_mExact = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
cs_kmeans_tSNE_lr200_mExact_per10 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per10)
cs_kmeans_tSNE_lr200_mExact_per15 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per15)
cs_kmeans_tSNE_lr200_mExact_per20 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per20)
cs_kmeans_tSNE_lr200_mExact_per40 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per40)
cs_kmeans_tSNE_lr200_mExact_per50 = metrics.completeness_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per50)

cs = [cs_init,
      cs_kmeans_tSNE_lr100,
      cs_kmeans_tSNE_lr100_per10,
      cs_kmeans_tSNE_lr100_per15,
      cs_kmeans_tSNE_lr100_per20,
      cs_kmeans_tSNE_lr200,
      cs_kmeans_tSNE_lr200_mExact,
      cs_kmeans_tSNE_lr200_mExact_per10,
      cs_kmeans_tSNE_lr200_mExact_per15,
      cs_kmeans_tSNE_lr200_mExact_per20,
      cs_kmeans_tSNE_lr200_mExact_per40,
      cs_kmeans_tSNE_lr200_mExact_per50
      ]

# V-measure Score
#vms_kmeans_PCA = metrics.v_measure_score(true_labels_int, x_kmeans_PCA)
#vms_kmeans_KernelPCA_cosine = metrics.v_measure_score(true_labels_int, x_kmeans_KernelPCA_cosine)
#vms_kmeans_KernelPCA_rbf = metrics.v_measure_score(true_labels_int, x_kmeans_KernelPCA_rbf)
#vms_kmeans_isomap = metrics.v_measure_score(true_labels_int, x_kmeans_isomap)
vms_kmeans_tSNE_lr100 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr100)
vms_kmeans_tSNE_lr100_per10 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr100_per10)
vms_kmeans_tSNE_lr100_per15 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr100_per15)
vms_kmeans_tSNE_lr100_per20 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr100_per20)
vms_kmeans_tSNE_lr200 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr200)
vms_kmeans_tSNE_lr200_mExact = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
vms_kmeans_tSNE_lr200_mExact_per10 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per10)
vms_kmeans_tSNE_lr200_mExact_per15 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per15)
vms_kmeans_tSNE_lr200_mExact_per20 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per20)
vms_kmeans_tSNE_lr200_mExact_per40 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per40)
vms_kmeans_tSNE_lr200_mExact_per50 = metrics.v_measure_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per50)

vms = [vms_init,
      vms_kmeans_tSNE_lr100,
      vms_kmeans_tSNE_lr100_per10,
      vms_kmeans_tSNE_lr100_per15,
      vms_kmeans_tSNE_lr100_per20,
      vms_kmeans_tSNE_lr200,
      vms_kmeans_tSNE_lr200_mExact,
      vms_kmeans_tSNE_lr200_mExact_per10,
      vms_kmeans_tSNE_lr200_mExact_per15,
      vms_kmeans_tSNE_lr200_mExact_per20,
      vms_kmeans_tSNE_lr200_mExact_per40,
      vms_kmeans_tSNE_lr200_mExact_per50
      ]

# Fowlkes-Mallows Score
#fms_kmeans_PCA = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_PCA)
#fms_kmeans_KernelPCA_cosine = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_KernelPCA_cosine)
#fms_kmeans_KernelPCA_rbf = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_KernelPCA_rbf)
#fms_kmeans_isomap = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_isomap)
fms_kmeans_tSNE_lr100 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr100)
fms_kmeans_tSNE_lr100_per10 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr100_per10)
fms_kmeans_tSNE_lr100_per15 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr100_per15)
fms_kmeans_tSNE_lr100_per20 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr100_per20)
fms_kmeans_tSNE_lr200 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr200)
fms_kmeans_tSNE_lr200_mExact = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr200_mExact)
fms_kmeans_tSNE_lr200_mExact_per10 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per10)
fms_kmeans_tSNE_lr200_mExact_per15 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per15)
fms_kmeans_tSNE_lr200_mExact_per20 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per20)
fms_kmeans_tSNE_lr200_mExact_per40 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per40)
fms_kmeans_tSNE_lr200_mExact_per50 = metrics.fowlkes_mallows_score(true_labels_int, x_kmeans_tSNE_lr200_mExact_per50)

fms = [fms_init,
      fms_kmeans_tSNE_lr100,
      fms_kmeans_tSNE_lr100_per10,
      fms_kmeans_tSNE_lr100_per15,
      fms_kmeans_tSNE_lr100_per20,
      fms_kmeans_tSNE_lr200,
      fms_kmeans_tSNE_lr200_mExact,
      fms_kmeans_tSNE_lr200_mExact_per10,
      fms_kmeans_tSNE_lr200_mExact_per15,
      fms_kmeans_tSNE_lr200_mExact_per20,
      fms_kmeans_tSNE_lr200_mExact_per40,
      fms_kmeans_tSNE_lr200_mExact_per50
      ]



# Plot the accuracy results in a bar plot
score_names = ['ars', 'amis', 'hs', 'cs', 'vms', 'fms']
scores = [ars, amis, hs, cs, vms, fms]
clustering_methods = ['Original', 't-SNE\n(lr=100)', 't-SNE\n(lr=100\nper=10)',
                      't-SNE\n(lr=100\nper=15)', 't-SNE\n(lr=100\nper=20)',
                      't-SNE\n(lr=200)', 't-SNE\n(lr=200,\nm=exact)',
                      't-SNE\n(lr=200,\nm=exact\nper=10)',
                      't-SNE\n(lr=200,\nm=exact\nper=15)',
                      't-SNE\n(lr=200,\nm=exact\nper=20)',
                      't-SNE\n(lr=200,\nm=exact\nper=40)',
                      't-SNE\n(lr=200,\nm=exact\nper=50)']
utils.plot_cluster_evaluation_scores(scores, score_names, clustering_methods)

#################################

# Experiments with multi-labeled images
# Dataset by grassias

test_folder = os.path.join(os.getcwd(), 'datasets', 'grassias')

# Read the test images filenames
test_images = os.listdir(test_folder)
test_images_paths = [os.path.join(test_folder, x) for x in test_images]

# Read the classes
with open(os.path.join(train_folder, 'selected_classes.txt'), 'r') as f:
    selected_classes = f.read().rstrip('\n').split('\n')

# Classify test images and show results
prob_vectors_multi = []
i = 1
fig_i = 1
fig = plt.figure(fig_i)

for x in range(12, len(test_images_paths)):
    print('\nClassifying image ' + test_images[x])
    res = visual_recognition.classify(open(test_images_paths[x], 'rb'),
                                      classifier_ids=classifier_id,
                                      threshold=0.0)
    print('Classified image ' + test_images[x])

    res = res['images'][0]['classifiers'][0]['classes']
    prob_vector = np.asarray([x['score'] for x in res])
    prob_vectors_multi.append(prob_vector)
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
prob_vectors_multi = np.asarray(prob_vectors_multi)

plt.show()

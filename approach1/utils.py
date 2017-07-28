# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import cv2
from random import sample
from shutil import copy2, make_archive, unpack_archive
import urllib3
from matplotlib import offsetbox
import matplotlib.pyplot as plt


def unpickle(file):
    '''
    Unpickles a given pickle file and returns its contents as a dictionary.
    '''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_folders(labels_meta, data_folder):
    '''
    Creates a separate folder for each label contained in the 'labels_meta'
    parameter.
    '''
    for class_label in labels_meta:
        try:
            os.mkdir(os.path.join(data_folder, class_label))
        except:
            pass


def unpickle_and_store_images(data_folder, labels_meta):
    '''
    Reads the data and stores each image in the respective folder in .jpg
    format.
    '''
    for i_batch in range(1, 6):
        batch = unpickle(os.path.join(data_folder, 'data_batch_' + str(i_batch)))
        for i in range(len(batch[b'labels'])):
            image_data = batch[b'data'][i].reshape((3, 32, 32))
            image_data = np.rollaxis(image_data, 0, 3)
            cv2.imwrite(os.path.join(data_folder,
                                     labels_meta[batch[b'labels'][i]],
                                     batch[b'filenames'][i].decode('utf8')),
                        image_data)


def compress_folders(data_folder, labels_meta):
    '''
    Compresses every folder whose name is included in 'labels_meta' into
    a .zip file and stores it in the same directory.
    The initial folders must be inside 'data_folder'.
    '''
    for class_label in labels_meta:
        make_archive(class_label, 'zip', root_dir=data_folder,
                     base_dir=class_label)


def uncompress_folders(data_folder, dest_folder):
    '''
    Uncompresses all compressed files inside data_folder. The compressed files
    must be named as 'something-label', where 'label' is the class label of the
    files. The uncompressed files are stored inside folders named with the
    respective 'label'.

    Intended for use at the ImageNet dataset, where the archives are of the
    form: imagenetCode-label.tar
    '''
    files = os.listdir(data_folder)
    for file in files:
        file_label = file.split('.')[0].split('-')[1]
        unpack_archive(os.path.join(data_folder, file),
                       extract_dir=os.path.join(dest_folder, file_label + '_all'))


def select_random_images(selected_classes, data_folder, train_folder,
                         number_of_images, test_folder=None,
                         number_of_test_images=10):
    '''
    Randomly selects a number of images from each class in
    'selected_classes' and copies them in separate folders in order to form
    a training set. If test_folder is specified, then this function also
    selects another subset of images (disjoint from those selected for
    training) and stores them all inside the test_folder.

    Parameters
    ----------
    selected_classes: list of str
        The labels of the selected classes.
    data_folder: str
        The dataset folder.
    train_folder: str
        The base folder for the training set to be created.
    number_of_images: int
        The number of images to select from each class.
    test_folder: int, default None
        The folder inside which to store the test set.
    number_of_test_images: int, default 10
        The number of test images to be selected from each class.
    '''
    for cl in selected_classes:
        try:
            os.mkdir(os.path.join(train_folder, cl))
        except:
            pass
        imagenames = os.listdir(os.path.join(data_folder, cl + '_all'))
        indices = sample(range(len(imagenames)),
                         number_of_images)
        for ind in indices:
            copy2(os.path.join(data_folder, cl+'_all', imagenames[ind]),
                  os.path.join(train_folder, cl, imagenames[ind]))

        if test_folder is not None:
            indices_test = sample(set(range(len(imagenames))) - set(indices))
            for ind in indices_test:
                copy2(os.path.join(data_folder, cl+'_all', imagenames[ind]),
                      os.path.join(test_folder, imagenames[ind]))


def download_save_image(url, test_folder):
    '''
    Downloads the image contained in the given URL and stores it
    inside the test data folder.
    '''
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)

    image_name = url.split('/')[-1]
    with open(os.path.join(test_folder, image_name), 'wb') as out:
        while True:
            data = r.read()
            if not data:
                break
            out.write(data)

    r.release_conn()
    return os.path.join(test_folder, image_name)


def create_collection(selected_classes, data_folder, collection_folder,
                      num_of_images):
    '''
    Randomly selects a number of images from a given data folder and
    stores them in a different destination folder.
    '''
    for cl in selected_classes:
        imagenames = os.listdir(os.path.join(data_folder, cl))
        indices = sample(range(len(imagenames)),
                         num_of_images)
        for ind in indices:
            copy2(os.path.join(data_folder, cl, imagenames[ind]),
                  os.path.join(collection_folder, imagenames[ind]))


def plot_embedding(X, imagepaths, collection_size, title=None,
                   test_image=False, image_size=42):
    '''
    Plots the images in the embedding space. The test image has red border.

    X: ndarray of float
        The distance matrix (2x2 for the 2D visualization).
    imagepaths: list of str
        A list containing all paths to the images.
    collection_size: int
        The number of images inside the collection.
    test_image: bool, default False
        If True, then we assume that the last vector in X corresponds to a
        test image, so we annotate it with a red box and the others with a
        black one. Otherwise, we annotate all images with a black box.
    image_size: int, default 42
        The final size of the thumbnails to be shown in the plot will be
        (image_size, image_size).

    Kudos to georgeretsi for the code.
    '''
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    #for i in range(X.shape[0]):
    #    plt.text(X[i, 0], X[i, 1], str(names[i]),fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            #dist = np.sum((X[i] - shown_images) ** 2, 1)
            #if np.min(dist) < 4e-3:
                # don't show points that are too close
            #    continue
            shown_images = np.r_[shown_images, [X[i]]]
            timg = cv2.imread(imagepaths[i])
            timg = timg[:, :, ::-1]  # BGR -> RGB
            timg = cv2.resize(timg, dsize=(image_size, image_size))
            if (i == X.shape[0] - 1) and (test_image == True):
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(timg, cmap=plt.cm.gray_r), X[i], bboxprops=dict(edgecolor='red'))
            else:
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(timg, cmap=plt.cm.gray_r), X[i], bboxprops=dict(edgecolor='black'))
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show(block=True)


def get_labels_imagenet(imagenames):
    '''
    Extracts the label from the image names according to ImageNet convention.
    '''
    synset = {'n01503061': 'bird', 'n02084071': 'dog', 'n02121808': 'cat',
              'n02691156': 'airplane', 'n02958343': 'car'}
    return [synset[x.split('_')[0]] for x in imagenames]


def get_labels_random(imagenames):
    '''
    Extracts the labels from the given image names.
    Naming should follow the format:
        label1_label2_..._labelx number.file_format
    where number is optional and is used only in case of multiple images with
    the same labels.
    '''
    all_labels = []
    for imagename in imagenames:
        labels = imagename.split('.')[0].split(' ')[0].split('_')
        all_labels.append(labels)

    return all_labels


def plot_cluster_evaluation_scores(scores, score_names, clustering_methods):
    '''
    Plots the given evaluation scores in a grouped bar plot.
    '''
    n = len(score_names)

    margin = 0.05  # Space between groups of data
    width = (1.-2.*margin)/n
    ind = np.arange(len(clustering_methods))  # the x locations of the groups

    fig, ax = plt.subplots()

    for i in range(n):
        plt.bar(ind+margin+(i*width), scores[i], width)

    ax.set_ylabel('Score')
    ax.set_title('Cluster evaluation scores')
    ax.set_xticks([i+(3.5 * width) for i in ind])
    ax.set_xticklabels(clustering_methods)
    plt.legend(score_names)
    plt.grid()

    plt.show()

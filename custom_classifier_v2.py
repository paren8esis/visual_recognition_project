# -*- coding: utf-8 -*-

# Wrap code for the Watson Visual Recognition classifier.
#
# API: https://www.ibm.com/watson/developercloud/visual-recognition/api/v3/#classify_an_image

import os
from watson_developer_cloud import VisualRecognitionV3
import cv2
import operator
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import utils


class CustomClassifier:
    '''
    Attributes
    ----------
    api_key
    version
    visual_recognition
    classifier_id
    classifier_status
    train_folder
    test_folder
    selected_classes

    Methods
    -------
    
    '''

    def __init__(self, api_key, version):
        self.api_key = api_key
        self.version = version
        self.visual_recognition = VisualRecognitionV3(version, api_key=api_key)
        self.classifier_id = ''
        self.classifier_status = ''
        self.train_folder = ''
        self.test_folder = ''
        self.selected_classes = ''

    def get_classifiers(self):
        '''
        Prints all classifiers that have been created so far.
        '''
        print("\nList of classifiers:")
        classifiers_list = self.visual_recognition.list_classifiers()
        print(classifiers_list)

    def update_classifier_info(self):
        '''
        Loads all necessary info about the classifier currently in our
        workspace.
        '''
        classifiers_list = self.visual_recognition.list_classifiers()
        self.classifier_id = classifiers_list['classifiers'][0]['classifier_id']
        self.classifier_status = classifiers_list['classifiers'][0]['status']
        classifier_details = self.visual_recognition.get_classifier(self.classifier_id)
        self.selected_classes = [x['class'] for x in classifier_details['classes']]

        print("Updated!")

    def get_classifier_id(self):
        '''
        Returns the id of the classifier currently in our workspace.
        '''
        classifiers_list = self.visual_recognition.list_classifiers()
        self.classifier_id = classifiers_list['classifiers'][0]['classifier_id']
        return self.classifier_id

    def get_classifier_status(self):
        '''
        Returns the status of the classifier currently in our workspace.
        '''
        classifiers_list = self.visual_recognition.list_classifiers()
        classifier_status = classifiers_list['classifiers'][0]['status']
        return classifier_status

    def get_classifier_details(self):
        '''
        Returns more details about the classifier.
        '''
        details = self.visual_recognition.get_classifier(self.classifier_id)
        print(details)

    def delete_classifier(self, classifier_id):
        '''
        Deletes the classifier that has the given id.
        '''
        self.visual_recognition.delete_classifier(classifier_id=classifier_id)
        self.classifier_id = ''
        print('Classifier deleted!')

    def create_classifier(self, train_data, name):
        '''
        Creates a new classifier trained with the given data.
        We assume that the train_data is a list of .zip filenames, and that
        every filename represents the corresponding class.

        Training limitations:
            1) Maximum of 10,000 images or 100 MB per .zip file
            2) Minimum of 10 images per .zip file
            3) Maximum of 256 MB per training call
            4) Minimum recommended size of image is 32 x 32 pixels

        If these limitations are not met, an error '104: Connection reset
        by peer' is raised.

        WARNING: this code is highly insecure. Data input by user is not
        sanitized and can pose serious security threats. Use with caution.
        '''
        command = "with "
        classnames = []
        for zipfile in train_data:
            command += "open('" + zipfile + "', 'rb') as " + \
                zipfile.split('.')[0] + ", "
            classnames.append(zipfile.split('.')[0])
        command = command[:-2]
        command += ": classifier = self.visual_recognition.create_classifier('" + \
            name + "', "
        for classname in classnames:
            command += classname + "_positive_examples=" + classname + ", "
        command = command[:-2]
        command += ")"

        exec(command)

        classifiers_list = self.visual_recognition.list_classifiers()
        self.classifier_id = classifiers_list['classifiers'][0]['classifier_id']

        self.selected_classes = [x.split('.')[0] for x in train_data]

    def set_train_folder(self, new_path):
        '''
        Sets the path of the training data.
        The folder must be inside our current working directory.
        '''
        self.train_folder = os.path.join(os.getcwd(), new_path)

    def set_test_folder(self, new_path):
        '''
        Sets the path of the testing data.
        The folder must be inside our current working directory.
        '''
        self.test_folder = os.path.join(os.getcwd(), new_path)

    def load_test_images(self):
        '''
        Reads the images in the test set and returns a list of the image names,
        a list of the full image paths and the class labels.
        '''
        # Read the test images filenames
        test_images = os.listdir(self.test_folder)
        test_images_paths = [os.path.join(self.test_folder, x) for x in test_images]

        # Read the classes
        with open(os.path.join(self.train_folder, 'selected_classes.txt'), 'r') as f:
            selected_classes = f.read().rstrip('\n').split('\n')

        return test_images_paths, selected_classes

    def classify(self, test_images_paths, selected_classes,
                 plot_results=True, verbose=True):
        '''
        Performs classification over the test images and, if needed, plots the
        results.
        '''
        classifier_details = self.visual_recognition.get_classifier(self.classifier_id)
        self.selected_classes = [x['class'] for x in classifier_details['classes']]

        test_images = [os.path.basename(x) for x in test_images_paths]
        prob_vectors = []
        i = 1
        if plot_results:
            fig_i = 1
            fig = plt.figure(fig_i)

        for x in range(len(test_images_paths)):
            if verbose:
                print('\nClassifying image ' + test_images[x])
            res = self.visual_recognition.classify(open(test_images_paths[x], 'rb'),
                                                   classifier_ids=self.classifier_id,
                                                   threshold=0.0)
            if verbose:
                print('Classified image ' + test_images[x])

            res = res['images'][0]['classifiers'][0]['classes']
            self.selected_classes = [x.split('.')[0] for x in train_data]
            prob_vector = np.asarray([x['score'] for x in res])
            prob_vectors.append(prob_vector)
            res_sorted = sorted(res, key=operator.itemgetter('score'),
                                reverse=True)

            if plot_results:
                if ((i-1) % 8 == 0) and (i != 1):
                    fig_i += 1
                    fig = plt.figure(fig_i)
                    i = 1
            test_image_arr = cv2.imread(test_images_paths[x])

            if plot_results:
                plt.subplot(240 + i)
                plt.imshow(test_image_arr[:, :, ::-1])
                plt.axis('off')
                plt.subplot(240 + i + 1)
                plt.bar(range(len(selected_classes)), [x['score'] for x in res_sorted])
                plt.xticks(range(len(selected_classes)),
                           [x['class'] for x in res_sorted],
                           rotation=65)
            #    plt.text(0, 0, ''.join([x['class'] + ': ' + str(x['score']) + '\n' for x in res_sorted]))
                plt.tight_layout()

            i += 2
        prob_vectors = np.asarray(prob_vectors)

        if plot_results:
            plt.show()

        np.save('prob_vectors', prob_vectors)
        return prob_vectors

    def get_true_test_labels(self, test_images_paths, selected_classes,
                             imagenet=True):
        '''
        Returns the true labels of the test images both in string and int
        format.
        '''
        test_images = [os.path.basename(x) for x in test_images_paths]
        if imagenet:
            true_labels = utils.get_labels_imagenet(test_images)
        else:
            pass
        labels_replace = {key: value for (key, value) in [(x, y) for x, y in zip(selected_classes, range(len(selected_classes)))]}
        true_labels_int = [labels_replace[x] for x in true_labels]

        return true_labels, true_labels_int

    def create_base_collection(self, num_images_per_class=10):
        '''
        Chooses a subset of the training set in order to visualize the
        resulting classification of the test image with them.
        '''
        all_indices = []
        collection_paths = []
        for cl in self.selected_classes:
            imagenames = os.listdir(os.path.join(self.train_folder, cl))
            indices = sample(range(len(imagenames)), num_images_per_class)
            all_indices.append(indices)
            collection_paths += [os.path.join(self.train_folder,
                                              cl,
                                              imagenames[i]) for i in indices]

        return collection_paths

    def visualize_results(self, train_set_prob, train_set_paths,
                          test_set_prob=None, test_set_paths=None,
                          perplexity=15, learning_rate=100, image_size=42):
        '''
        Runs t-SNE on the given images and visualizes the result in a 2D plot.
        '''
        model_tSNE = TSNE(n_components=2, random_state=0,
                          learning_rate=learning_rate, perplexity=perplexity)

        if test_set_prob is not None:
            x_tSNE = model_tSNE.fit_transform(np.vstack((train_set_prob,
                                                         test_set_prob)))
            utils.plot_embedding(x_tSNE,
                                 np.vstack((train_set_paths, test_set_paths)),
                                 len(train_set_paths) + len(test_set_paths),
                                 image_size=image_size, test_image=True)
        else:
            x_tSNE = model_tSNE.fit_transform(train_set_prob)
            utils.plot_embedding(x_tSNE, train_set_paths, len(train_set_paths),
                                 image_size=image_size, test_image=False)

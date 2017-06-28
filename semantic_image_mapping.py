# -*- coding: utf-8 -*-

# Wrap code for the Watson Visual Recognition classifier and the Semantic
# Image Mapping.
# Throughout the code we assume that a free API key is used, so there can only
# be one classifier in the workspace.
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
    api_key : str
        The api key to be used for Watson services.
    version : str
        The api version to be used for Watson services.
    visual_recognition : VisualRecognition() instance
        An instance of Watson's VisualRecognition() class.
    classifier_id : str
        The current classifier's id.
    classifier_status : str
        The current classifier's status.
    train_folder : str
        The folder path containing the training set.
    test_folder : str
        The folder path containing the test set.
    selected_classes : list of str
        The classes on which the classifier has been trained.

    Methods
    -------
    get_classifiers()
        Prints a list of all the classifiers currently in the workspace
        (due to the free api key, we are expecting at most 1 classifier).
    get_classifier_id()
        Returns the id of the current classifier.
    get_classifier_status()
        Returns the status of the current classifier.
    get_classifier_details()
        Returns some details about the current classifier.
    delete_classifier()
        Deletes the current classifier.
    create_classifier(train_data, name)
        Creates a new classifier with the given name and data.
    set_train_folder(new_path)
        Sets the folder path of the training set to be used.
    set_test_folder(new_path)
        Sets the folder path of the test set to be used.
    load_test_images()
        Returns a list of the paths of the test images contained in 
        test folder.
    classify(test_images_paths, plot_results=True, verbose=True)
        Classifies the images in the given paths and plots the results.
    create_base_collection(self, num_images_per_class=10)
        Creates a base collection from the training set and returns a list of
        the paths.
    visualize_results(self, train_set_prob, train_set_paths,
                      test_set_prob=None, test_set_paths=None,
                      perplexity=15, learning_rate=100, image_size=42)
        Runs the t-SNE algorithm on the given images and plots a 2D plot of the
        results.
    '''

    def __init__(self, api_key, version):
        '''
        Parameters
        ----------
        api_key : str
            The api key to be used
        version : str
            The version to be used

        Notes
        -----
        Updates the api_key, version, visual_recognition, classifier_id,
        classifier_status, selected_classes, train_folder and test_folder
        attributes.
        '''
        self.api_key = api_key
        self.version = version
        self.visual_recognition = VisualRecognitionV3(version, api_key=api_key)

        # Check if there are any classifiers in the workspace
        classifiers_list = self.visual_recognition.list_classifiers()
        if classifiers_list['classifiers'] == []:
            self.classifier_id = ''
            self.classifier_status = ''
            self.selected_classes = ''
        else:
            self.classifier_id = classifiers_list['classifiers'][0]['classifier_id']
            self.classifier_status = classifiers_list['classifiers'][0]['status']
            classifier_details = self.visual_recognition.get_classifier(self.classifier_id)
            self.selected_classes = [x['class'] for x in classifier_details['classes']]
        self.train_folder = ''
        self.test_folder = ''

    def get_classifiers(self):
        '''
        Prints all classifiers that have been created so far.
        '''
        print("\nList of classifiers:")
        classifiers_list = self.visual_recognition.list_classifiers()
        print(classifiers_list)

    def get_classifier_id(self):
        '''
        Returns the id of the classifier currently in our workspace.

        Output
        ------
        classifier_id : str
            The id of the current classifier.

        Notes
        -----
        Updates the classifier_id attribute.
        '''
        classifiers_list = self.visual_recognition.list_classifiers()
        self.classifier_id = classifiers_list['classifiers'][0]['classifier_id']
        return self.classifier_id

    def get_classifier_status(self):
        '''
        Returns the status of the classifier currently in our workspace.

        Output
        ------
        classifier_status : str
            The status of the current classifier.

        Notes
        -----
        Updates the classifier_status attribute.
        '''
        classifiers_list = self.visual_recognition.list_classifiers()
        self.classifier_status = classifiers_list['classifiers'][0]['status']
        return self.classifier_status

    def get_classifier_details(self):
        '''
        Returns more details about the classifier.

        Notes
        -----
        Updates the selected_classes attribute.
        '''
        details = self.visual_recognition.get_classifier(self.classifier_id)
        print(details)
        self.selected_classes = [x['class'] for x in details['classes']]

    def delete_classifier(self):
        '''
        Deletes the classifier currently in the workspace.

        Notes
        -----
        Updates the classifier_id, classifier_status and selected_classes 
        attributes.
        '''
        delete_keep = input('Are you sure you want to delete the classifier? (y/n) ')

        if delete_keep in ['Y', 'y']:
            self.visual_recognition.delete_classifier(classifier_id=self.classifier_id)
            self.classifier_id = ''
            self.classifier_status = ''
            self.selected_classes = ''
            print('Classifier deleted!')
        else:
            print('Deletion aborted')

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

        Parameters
        ----------
        train_data : list of str
            A list containing the names of the compressed files to be used in
            the classifier's training.
        name : str
            The name of the new classifier.

        Notes
        -----
        Updates the classifier_id, classifier_status and selected_classes 
        attributes.
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
        self.classifier_status = classifiers_list['classifiers'][0]['status']

        self.selected_classes = [x.split('.')[0] for x in train_data]

    def set_train_folder(self, new_path):
        '''
        Sets the path of the training data.
        The folder must be inside our current working directory.

        Parameters
        ----------
        new_path : str
            The new path to be used for the training folder.
        '''
        self.train_folder = os.path.join(os.getcwd(), new_path)

    def set_test_folder(self, new_path):
        '''
        Sets the path of the testing data.
        The folder must be inside our current working directory.

        Parameters
        ----------
        new_path : str
            The new path to be used for the test folder.
        '''
        self.test_folder = os.path.join(os.getcwd(), new_path)

    def load_test_images(self):
        '''
        Reads the images in the test set and returns a list of the image names,
        a list of the full image paths and the class labels.

        Output
        ------
        test_images_paths : list of str
            A list containing the full paths of the test images.
        '''
        # Read the test images filenames
        test_images = os.listdir(self.test_folder)
        test_images_paths = [os.path.join(self.test_folder, x) for x in test_images]

        return test_images_paths

    def classify(self, test_images_paths, plot_results=True, verbose=True):
        '''
        Performs classification over the test images and, if needed, plots the
        results.

        Parameters
        ----------
        test_images_paths : list of str
            A list containing the paths of the test images.
        plot_results : bool, default True
            If True, then the results of the classification are plotted for
            each image.
        verbose : bool, default True
            If True, then a message appears every time a new image is being
            classified.

        Output
        ------
        prob_vectors : ndarray of float
            An ndarray containing the probability vectors of every image
            classified.
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
                plt.bar(range(len(self.selected_classes)), [x['score'] for x in res_sorted])
                plt.xticks(range(len(self.selected_classes)),
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

    def create_base_collection(self, num_images_per_class=10):
        '''
        Chooses a subset of the training set in order to visualize the
        resulting classification of the test image with them.

        Parameters
        ----------
        num_images_per_class : int, default 10
            The number of images to be chosen from each class.

        Output
        ------
        collection_paths : list of str
            A list containing the full paths of the images chosen for the
            base collection.
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
        If a test set is given (by passing a value for the test_set_prob
        parameter), then the images of the test set are annotated with red
        boxes.

        Parameters
        ----------
        train_set_prob : ndarray of float
            An ndarray containing the probability vectors of each classified
            image.
        train_set_paths : list of str
            A list containing the paths of the classified images.
        test_set_prob : ndarray of float, default None
            An ndarray containing the probability vectors of each classified
            test image.
        test_set_paths : list of str
            A list containing the paths of the classified test images.
        perplexity : int, default 15
            The perplexity parameter for the t-SNE algorithm.
        learning_rate : int, default 100
            The learning rate parameter for the t-SNE algorithm.
        image_size : int, default 42
            The thumbnail size for the plot.
        '''
        model_tSNE = TSNE(n_components=2, random_state=0,
                          learning_rate=learning_rate, perplexity=perplexity)

        if test_set_prob is not None:
            x_tSNE = model_tSNE.fit_transform(np.vstack((train_set_prob,
                                                         test_set_prob)))
            if type(train_set_paths) == list:
                train_set_paths = np.array(train_set_paths)
            if len(train_set_paths.shape) == 1:
                train_set_paths = train_set_paths[:, np.newaxis]

            if type(test_set_paths) == list:
                test_set_paths = np.array(test_set_paths)
            if len(test_set_paths.shape) == 1:
                test_set_paths = test_set_paths[:, np.newaxis]
            utils.plot_embedding(x_tSNE,
                                 np.squeeze(np.vstack((train_set_paths, test_set_paths))),
                                 train_set_paths.shape[0] + test_set_paths.shape[0],
                                 image_size=image_size, test_image=True)
        else:
            x_tSNE = model_tSNE.fit_transform(train_set_prob)
            utils.plot_embedding(x_tSNE, train_set_paths, len(train_set_paths),
                                 image_size=image_size, test_image=False)


if __name__ == '__main__':

    api_key = '4fa5700177e476ed183d077aa826ea1f6cc8a383'
    version = '2016-05-20'

    cc = CustomClassifier(api_key, version)

    collection_paths = np.load('collection_paths.npy')
    collection_prob_vectors = np.load('collection_prob_vectors.npy')
    test_prob_vectors = np.load('test_prob_vectors.npy')

    cc.set_train_folder('datasets/ImageNet')
    cc.set_test_folder('datasets/test_set')
    test_images_paths = cc.load_test_images()

    cc.visualize_results(collection_prob_vectors,
                         collection_paths)

    cc.visualize_results(collection_prob_vectors,
                         collection_paths,
                         test_set_prob=test_prob_vectors[1:2],
                         test_set_paths=test_images_paths[1:2],
                         perplexity=7)
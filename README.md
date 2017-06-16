# README #

### Idea ###

1. Train Watson Visual Classifier with the [ImageNet dataset](http://www.image-net.org/).
2. Choose a few random images from the web as test set (here using grassias' test set).
3. Use the classifier to compute their probability vectors.
4. Visualize the test set using a dimensionality reduction technique (e.g. t-SNE) on their probability vectors.

### Notes ###

First run the data preparation script (*imagenet_data_preparation.py*) to prepare the datasets. Then run the file *custom_classifier.py* to do the classification and visualization of the results. The file *utils.py* contains a few useful routines used by all other scripts.

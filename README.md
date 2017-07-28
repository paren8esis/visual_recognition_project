# README #

## Approach 1 ##
### Idea ###

1. Train Watson Visual Classifier with the [ImageNet dataset](http://www.image-net.org/).
2. Choose a few random images from the web as test set (here using test set from either ImageNet or Google Images Search).
3. Use the classifier to compute their probability vectors.
4. Visualize the test set using a dimensionality reduction technique (i.e. t-SNE) on their probability vectors.

### Notes ###

First run the data preparation script (*imagenet_data_preparation.py*) to prepare the datasets. Then use the file *semantic_image_mapping.py* to do the classification and visualization of the results. The file *utils.py* contains a few useful routines used by all other scripts.

## Approach 2 ##
### Idea ###

1. Create a collection of images using the beta Similarity Search service of Watson Visual Recognition.
2. Create a distance matrix by calculating pairwise distances of the existing images in the collection.
3. Calculate the embeddings of the initial images using a manifold embedding technique and the extracted distance matrix.    
4. Visualize the extracted embeddings and evaluate their capability of effectively form clusters (using k-means).

### Notes ###
The file *watson_collections.py* includes all the function for creating and handling collections, as well as the functions for calculating the distance matrix. The file *eval_utils.py* contains routines for vizualization and evaluation of critical parameters (a basix example of this approach is implemented on "simple_example" function).

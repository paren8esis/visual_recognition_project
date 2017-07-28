from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
#from numpy.random import shuffle

def plot_embedding(X, imgnames , title=None):
    #K = 35
    #rn_values = range(len(imgnames))
    #shuffle(rn_values)
    #sample = rn_values[:K]
    
    #X = X[sample,:]
    #imgnames = imgnames[sample]
    
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    #for i in range(X.shape[0]):
    #    plt.text(X[i, 0], X[i, 1], str(imgnames[i]),fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            #dist = np.sum((X[i] - shown_images) ** 2, 1)
            #if np.min(dist) < 4e-3:
                # don't show points that are too close
            #    continue
            shown_images = np.r_[shown_images, [X[i]]]
	    timg = misc.imread(imgnames[i])
	    timg = np.array(misc.imresize(timg,(70,70,3))) 
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(timg, cmap=plt.cm.gray_r),X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show(block=True)
    #plt.savefig('/home/george/Dropbox/myfig.png')	


from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

from time import time
from sklearn.preprocessing import scale

from watson_collections import *

def bench_k_means(estimator, name, data, labels):
    t0 = time()
    #data = scale(data)
    estimator.fit(data)
    print('% 9s   %.2fs  %.3f   %.3f   %.3f   %.3f   %.3f '
          % (name, (time() - t0), 
             #estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_)))


def evaluate_embeddings(D,labels):

    estimators = [KMeans(init='k-means++', n_clusters=5, n_init=10)] #,AgglomerativeClustering(n_clusters=5),AgglomerativeClustering(n_clusters=5,linkage='average')]
    est_names = ['KMeans'] #,'wardAgglomerativeClustering','avgAgglomerativeClustering']
    for e in range(len(estimators)):
        print '!!----------------------------------!!'
        print est_names[e]
        estim = estimators[e]
        for i in range(2,6+1):
            
            print '--------------------------------------'
            print '#dim = '+str(i)
            
            model_t = TSNE(n_components=i, learning_rate = 100, perplexity = 10, method='exact')
            x = model_t.fit_transform(D) 
            bench_k_means(estim,name="tsne", data=x, labels=labels)
            
            model_i = Isomap(n_components=i)
            x = model_i.fit_transform(D) 
            bench_k_means(estim,name="isomap", data=x, labels=labels)
            
            model_l = LocallyLinearEmbedding(n_components=i)
            x = model_l.fit_transform(D) 
            bench_k_means(estim,name="lle", data=x, labels=labels)
        
def evaluate_similarity_function():

    #vr_instance = vr(version='2016-05-20', api_key='ca62a5844926baf007e5558a1d4c236dbccee838')
	
    walk_dir = './images'
    [names,labels] = find_images(walk_dir)
    collection_id = init_collection()
    fail_id = fill_collection(names,collection_id)
    mask = np.ones(len(names), dtype=bool)
    mask[fail_id] = False
    names = names[mask]
    labels = labels[mask]
    
    model_tsne = TSNE(n_components=2,  learning_rate = 100, perplexity = 20, method='exact')
    
    [D1,mask1] = create_distance_matrix(names,collection_id,100,1)
    names1 = names[mask1]
    labels1 = labels[mask1]
    x1 =  model_tsne.fit_transform(D1) 
    mask_p1 = np.zeros(len(names1), dtype=bool)
    for i in np.unique(labels1):
        tids = list(np.where(labels1==i)[0][1:8])
        mask_p1[tids] = True
    
    bench_k_means(KMeans(init='k-means++', n_clusters=5, n_init=10),name="tsne", data=x1, labels=labels1)
    plot_embedding(x1[mask_p1,:],names1[mask_p1])
    
    
    [D2,mask2] = create_distance_matrix(names,collection_id,100,2)
    names2 = names[mask2]
    labels2 = labels[mask2]
    x2 = model_tsne.fit_transform(D2) 

    mask_p2 = np.zeros(len(names2), dtype=bool)
    for i in np.unique(labels2):
        tids = list(np.where(labels2==i)[0][1:8])
        mask_p2[tids] = True
    
    bench_k_means(KMeans(init='k-means++', n_clusters=5, n_init=10),name="tsne", data=x2, labels=labels2)
    plot_embedding(x2[mask_p2,:],names2[mask_p2])


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
    
    
def change_limit():

    #vr_instance = vr(version='2016-05-20', api_key='ca62a5844926baf007e5558a1d4c236dbccee838')	
   
    walk_dir = './images'
    [names,labels] = find_images(walk_dir)
    collection_id = init_collection()
    fail_id = fill_collection(names,collection_id)
    mask = np.ones(len(names), dtype=bool)
    mask[fail_id] = False
    names = names[mask]
    labels = labels[mask]

    for k in [5,10,20,30,50,100]:
        [Dt,maskt] = create_distance_matrix(names,collection_id,k,2)
        #namest = names[maskt]
        labelst = labels[maskt]
        evaluate_embeddings(Dt,labelst)


def simple_example():
    
    #vr_instance = vr(version='2016-05-20', api_key='ca62a5844926baf007e5558a1d4c236dbccee838')	
    
    walk_dir = './images'
    [names,labels] = find_images(walk_dir)
    collection_id = init_collection()
    fail_id = fill_collection(names,collection_id)
    mask = np.ones(len(names), dtype=bool)
    mask[fail_id] = False
    names = names[mask]
    labels = labels[mask]
    [D,mask] = create_distance_matrix(names,collection_id,30,2)
    names = names[mask]
    labels = labels[mask]
    
    model_tsne = TSNE(n_components=2, random_state=0)
    x = model_tsne.fit_transform(D) 

    evaluate_embeddings(D,labels)
    
    # plot subset	 
    mask = np.zeros(len(names), dtype=bool)
    for i in np.unique(labels):
        tids = list(np.where(labels==i)[0][1:8])
        mask[tids] = True

    plot_embedding(x[mask,:],names[mask])

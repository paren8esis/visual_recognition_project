# -*- coding: utf-8 -*-

import os
from os.path import join
from watson_developer_cloud import VisualRecognitionV3 as vr
import numpy as np

vr_instance = vr(version='2016-05-20', api_key='ca62a5844926baf007e5558a1d4c236dbccee838')

def find_images(walk_dir):
    '''
    Find images of different labels (corresponding to different subfolders) according to parent folder 'walk_dir'.
    '''		

    names = []
    fnames = []
    str_labels = []
    for root, subdirs, files in os.walk(walk_dir):
        for filename in files:
            if filename.endswith(('.jpg','.png','.JPEG')):
                full_fname = join(root, filename)
                names.append(filename)
                fnames.append(full_fname)
                str_labels.append(root)
          
    [u, labels] = np.unique(str_labels, return_inverse=True)        
    return np.array(fnames), np.array(labels)

def init_collection():
    '''
    initialize empty collection while deleting previously existing ones.
    '''		

    previous_collections = vr_instance.list_collections()
    for p_collection in previous_collections['collections']:
        vr_instance.delete_collection(collection_id= p_collection['collection_id'])
    collection = vr_instance.create_collection('temp_collection')
    collection_id = collection['collection_id']

    return collection_id
    
def fill_collection_aux(tnames,collection_id):
    '''
    Check for Watson failure.
    '''		
    N = len(tnames)	
    fail_id = [] #np.zeros(N)		 
    for i in range(N):
        #print i
        #print tnames[i]
        try:
            I = open(tnames[i],'rb')    
            vr_instance.add_image(collection_id,I)
        except KeyboardInterrupt:
            print "Bye"
            return    
        except:
            #fail_id[i] = 1
            fail_id.append(i)
            print "watson @@: "+tnames[i]
            
    return np.array(fail_id)   

def fill_collection(fnames,collection_id):
    '''
    Fill created collection with the selected images (denoted by their path 'tnames').
    '''	
    tnames = fnames
    #while len(tnames) > 0 :
    #    fail_id = fill_collection_aux(tnames,collection_id) 
    #    tnames = tnames[fail_id]
    fail_id = fill_collection_aux(tnames,collection_id) 
    
    return fail_id
        
def create_distance_matrix(fnames,collection_id, K = 10, opt = 1):
    '''
    Create distance matrix from similarity score of similar images. 
    'K' is the number of neighbors and 'opt' is the distance function. 
    If opt == 1 f(s) = 1-s else if opt == 2 f(s) = -log(s).  
    '''		

    fnames = list(fnames)
    N= len(fnames)
    D = [[0 for i in range(N)] for j in range(N)]
    
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        #I = open(fnames[i],'rb');
        #report_similar = vr_instance.find_similar(collection_id,I,limit=K)
        try:
            I = open(fnames[i],'rb')    
            report_similar = vr_instance.find_similar(collection_id,I,limit=K)
            id1 = fnames.index(report_similar['image_file'])
            for similar_img in report_similar['similar_images']:
                id2 =  fnames.index(similar_img['image_file'])
                #D[id1][id2] = -np.log(similar_img['score'])#1-similar_img['score']  
                if opt == 1:
                    D[id1][id2] = 1-similar_img['score']   
                if opt == 2:
                    D[id1][id2] = -np.log(similar_img['score'])
        except KeyboardInterrupt:
            print "Bye"
            return    
        except:
            #fail_id[i] = 1
            mask[i] = False
            print "watson @@: "+fnames[i]
            
    D = np.array(D) 
    D = D[np.ix_(mask,mask)]   
    return D, mask

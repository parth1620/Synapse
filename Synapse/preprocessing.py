# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:10:47 2020

@author: Parth B Dhameliya
"""

'''incomplete'''

import numpy as np


def shuffle(X,y,seed = 0): 
        
        X = np.array(X)
        y = np.array(y)
        m1,n1=X.shape
        
        if y.ndim == 1:
            y = y.reshape(-1,1)
        else:
            pass
        
        data = np.hstack((X, y)) 
        np.random.seed(seed)
        np.random.shuffle(data)
        
        X = data[:, 0:n1] 
        y = data[:,n1:]
        
        return X,y.astype(int)

class Encoder:
    
    def get_one_hot_encode(self,targets):
        targets = np.array(targets)
        nb_classes = np.unique(targets)
        nb_classes.sort()
        res = np.eye(len(nb_classes))[np.array(targets).reshape(-1)]
        res = res.astype(int)
        return res
    
    def get_label_encode(self,targets):
        classes,targets = np.unique(targets,return_inverse = True)
        return targets
    
class Scaling:
    
    def normalization(self,X):
        mean = np.mean(X,axis = 0)
        std = np.std(X,axis = 0)
        X = (X - mean)/std
        return X
    

    
    
    






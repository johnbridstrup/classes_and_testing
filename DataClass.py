import numpy as np
import pandas as pd

class dataClass:

    def __init__(self, data=None, labels=None) -> None:
        if data is not None:
            if isinstance(data, list):
                self.data = np.array(data)
            elif isinstance(data, np.ndarray):
                self.data = data
        else:
            self.data = None

        if labels is not None:
            _,cols = self.data.shape
            if len(labels) != cols:
                raise IndexError('Must have same number of labels as data columns')
            if len(labels) != len(set(labels)):
                raise IndexError('Cannot contain duplicate labels')
            self.data_labels = labels
        else:
            self.data_labels = None

    def __getitem__(self, keys):

        if isinstance(keys,str):
            if self.data_labels is None:
                raise IndexError('No labels in data set')
            try:
                index = self.data_labels.index(keys)
            except:
                raise IndexError('{} is not a valid key'.format(keys))
            return self.data[:,index]

        else:
            try:
                return self.normalized_data[keys]
            except:
                return self.data[keys]

    def normalize(self):
        self.mu = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.normalized_data = self.data - self.mu.T
        self.normalized_data = self.normalized_data / self.std
        self.normalized = True

    def add_poly_feature(self, indices, label=None):
        if len(indices) < 2:
            raise IndexError('Polynomial has to be POLY')
        if label is not None:
            try:
                self.data_labels.append(label)
                if len(self.data_labels) != len(set(self.data_labels)):
                    raise IndexError('Duplicate label')
            except:
                raise IndexError('Labels havent been created for data')
        
        new_col = np.ones((self.data.shape[0],1))
        print(len(indices))
        for idx in indices:
            if idx >= len(self.data_labels):
                raise IndexError('Index, {}, is larger than number of features, {}.'.format(idx, len(self.data_labels)))
            new_col = np.multiply(new_col, self.data[:,idx,None])
            
        self.data = np.append(self.data,new_col,1)
        if self.normalized:
            # Can make this WAY more efficient but don't need to for the purposes of this
            self.normalize()
    

    

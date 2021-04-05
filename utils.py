import numpy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import copy


def unpickle(file, n=10000):
    """
    Parameters: 
    file (str): file address 
    num (int): number of datapoints in the file
    """
    import pickle
    import copy 

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    # let's make the Y (which is a K*n matrix of the one-hot representations of the label of each image)
    Y = np.zeros((10, n))
    zero_list = [0 for i in range(10)]
    i = 0
    for a in dict[b'labels']:
        '''
        right = str(bin(a+1)).split('b')[1]
        one_hot = np.array(list('0'*(10-len(right))+right))
        '''
        one_hot = copy.deepcopy(zero_list)
        one_hot[int(a)] = 1
        Y[:, i] = one_hot
        i += 1
        
        

    return dict[b'data'].T, Y, np.array(dict[b'labels'])




def normalize(data_):
    """
    This is used to normalize the data w.r.t mean and std: (x - mean) / std
    """
    data = np.copy(data_)
    shape = data.shape
    mean = np.mean(data, 1)
    std = np.std(data, 1)


    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - mean) / std

    return data
    

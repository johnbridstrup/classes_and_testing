from numpy.lib.arraysetops import isin
import sys
sys.path.append("..")
import numpy as np
from dataClass import DataClass

def test_normalize():
    """
    First real method test. Check that normalization is working properly for a simple test array
    with non-trivial mean and standard deviation
    """
    arr = np.array([[1,2],[10,50]])
    mu = arr.mean(axis=0)
    std = arr.std(axis=0)
    inst = DataClass(arr)
    norm_vec = arr - mu
    norm_vec = norm_vec / std
    inst.normalize()
    errors=[]

    if inst.mu[0] != mu[0] or inst.mu[1] != mu[1]:
        errors.append("wrong average")

    if inst.std[0] != std[0] or inst.std[1] != std[1]:
        errors.append("wrong std")

    for idx, val in enumerate(inst[:,:].ravel()):
        fail = False
        if val != norm_vec.ravel()[idx]:
            fail = True
        if fail:
            errors.append('incorrect normalization value')

    assert not errors, "errors occured:\t{}".format(errors)

def test_add_poly_feature():
    """
    This method should create polynomial features from un-normalized data.
    can take a label or autogenerate one (only if labels already exist).

    Method takes tuple of column indices (i, j, k, ...) and creates new feature 
    by multiplying x_i * x_j * x_k * ...
    """
    arr = np.array([[1,2],[10,50]])
    poly_arr = np.array([[1,2,1,2],[10,50,100,500]])
    inst = DataClass(arr,['data1','data2'])
    inst.normalize()

    poly1 = [0,0] # square first column
    poly2 = [0,1] # multiply first and second column

    errors=[]

    try:
        inst.add_poly_feature(poly1, 'data3')
        inst.add_poly_feature(poly2, 'data4')
    except Exception as e:
        # Any general error
        errors.append("{}".format(e))

    # New columns created
    try:
        x = inst[:,2]
        y = inst[:,3]
    except:
        errors.append("Not creating new columns")

    # Check new columns are labeled
    try:
        x = inst['data3']
        y = inst['data4']
    except Exception as e:
        errors.append("Labels: {}".format(e))

    # Compare new column values to expected
    value_flag = False
    try:
        for i, val in enumerate(inst.data[:,2]):
            if val != poly_arr[i,2]:
                value_flag= True
        for i, val in enumerate(inst.data[:,3]):
            if val != poly_arr[i,3]:
                value_flag= True
        if value_flag:
            errors.append("Polynomial values incorrect")
    except Exception as e:
        errors.append('Values: {}'.format(e))
    
    assert not errors, "errors occured:\t{}".format(errors)

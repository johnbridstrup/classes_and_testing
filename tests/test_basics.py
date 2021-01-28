from numpy.lib.arraysetops import isin
import pytest
import sys
sys.path.append("..")
import numpy as np
from DataClass import dataClass

def check_1d_array(arr, checkarr):
    out = True
    for idx, val in enumerate(arr):
        if val != checkarr[idx]:
            out = False

def test_construction_bare():
    inst = dataClass()
    assert not inst.data

def test_construction_array():
    arr = np.array([[1,1],[2,2]])
    inst = dataClass(arr)
    assert isinstance(inst.data, np.ndarray)

def test_construction_list():
    l = [[1,1],[2,2]]
    inst = dataClass(l)
    assert isinstance(inst.data, np.ndarray)

def test_index_lookup():
    arr = np.array([[1,1],[2,2]])
    inst = dataClass(arr)
    assert inst[1,1] == 2

def test_slices():
    arr = np.array([[1,1],[2,2]])
    inst = dataClass(arr)
    assert inst[1,:].size == 2

def test_label_basics():
    # Just make sure using string keys doesn't raise exceptions
    arr = np.array([[1,2,3],[10,20,30]])
    labels = ['data1','data2','data3']
    dup_labels = ['data1','data2','data1']
    fewer_labels = ['data1','data2']
    more_labels = ['data1','data2','data3','data4']
    inst = dataClass(arr, labels=labels)
    fail = False
    
    errors=[]
    # Check label lookup
    try:
        x = inst['data1']
    except IndexError:
        errors.append('Label lookup fail')

    # Check duplicates raises exception
    try:
        inst = dataClass(arr,labels=dup_labels)
        errors.append('Duplicates allowed')
    except IndexError:
        pass
    
    # too few labels
    try:
        inst = dataClass(arr, labels = fewer_labels)
        errors.append('Too few labels works')
    except IndexError:
        pass

    # too few labels
    try:
        inst = dataClass(arr, labels = fewer_labels)
        errors.append('Too few labels works')
    except IndexError:
        pass

    # check too many labels throws error 
    try:
        inst = dataClass(arr, labels = more_labels)
        errors.append('Too many labels works')
    except IndexError:
        pass

    assert not errors, "errors occured:\t{}".format(errors)


def test_labelling_data():
    arr = np.array([[1,2,3],[10,20,30]])
    labels = ['data1','data2','data3']
    
    

    errors = []

    ## Correct functioning first
    inst = dataClass(arr, labels=labels)

    # Check that labelling returns a 1D numpy array
    if not isinstance(inst['data1'],np.ndarray):
        errors.append('label not returning array')

    # Check that the data contained matches columns of input array
    label_fail = False
    for i,label in enumerate(labels):
        if inst[label][0] != arr[0,i] or inst[label][1] != arr[1,i]:
            label_fail = True
    if label_fail:
        errors.append('Columns dont match')
    
    
    assert not errors, "errors occured:\t{}".format(errors)
    
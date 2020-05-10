import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

test_data_dir = 'C:/Users/Robert/Desktop/College/Project/Data/ModelData/test'
train_data_dir = 'C:/Users/Robert/Desktop/College/Project/Data/ModelData/train'
genre = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

train_data = []
test_data = []

def create_train_data(): #create train_data[]:
    print('[+]: Starting...')
    print('[+]: Creating Training Data...')
    for g in genre:
        path = os.path.join(train_data_dir, g) # ../ModelData/test/blues
        tr_class_num = genre.index(g) # [genre] index vals as class vals, bluies = 0, classical = 1
        for img in os.listdir(path):
            tr_img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # convert to pixel vals, B&W
            train_data.append([tr_img_array, tr_class_num]) # [ [145,134...155], 1 ]
    create_test_data() 

def create_test_data(): #create test_data[]:
    print('[+]: Creating Test Data...')
    for g in genre:
        path = os.path.join(test_data_dir, g)
        ts_class_num = genre.index(g)
        for img in os.listdir(path):
            ts_img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            test_data.append([ts_img_array, ts_class_num])
    shuffle_data() 

def shuffle_data(): #shuffle data 
    print('[+]: Shuffling Data...')
    random.shuffle(train_data)
    random.shuffle(test_data)

    create_model_sets()

def create_model_sets():
    print('[+]: Creating Model Data Sets...')
    X_train = [] #feature
    X_test = [] 
    y_train = [] #label
    y_test = [] 
    # create datasets
    for features, label in train_data:
        X_train.append(features) #pixel vals
        y_train.append(label) #class vals
    for features, label in test_data:
        X_test.append(features)
        y_test.append(label)
    # convert to np.array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # normalise data
    X_train = (X_train/255) 
    X_test = (X_test/255)

    pickle_data(X_train, y_train, X_test, y_test)


def pickle_data(X_train, y_train, X_test, y_test):
    print('[+]: Pickling Data...')
    #pickle train data
    pickle_out = open('../Data/ModelData/X_train.pickle', 'wb')
    pickle.dump(X_train, pickle_out)
    pickle_out.close()
    pickle_out = open('../Data/ModelData/y_train.pickle', 'wb')
    pickle.dump(y_train, pickle_out)
    pickle_out.close()
    #pickle test data
    pickle_out = open('../Data/ModelData/X_test.pickle', 'wb')
    pickle.dump(X_test, pickle_out)
    pickle_out.close()
    pickle_out = open('../Data/ModelData/y_test.pickle', 'wb')
    pickle.dump(y_test, pickle_out)
    pickle_out.close()

    print(X_train)
    print('[+]: Done...')

create_train_data() 

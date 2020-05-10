import os
import random
import numpy as np
import tensorflow as tf
import pickle
from keras import backend as K
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import LSTM, Flatten
from keras.optimizers import Adam
import keras 
import matplotlib.pyplot as plt
import time
from tensorflow.keras.callbacks import TensorBoard


# naming convention ..
# # {network}-{LTSM}-{dense.layers}-{epochs}-{drop.out}


# setup
NAME = "RNN-1000-e.15-DO-0.5-{}".format(int(time.time()))
plot_name = os.path.join("C:/Users/Robert/Desktop/College/Project/Model/Plots/", NAME+".png")
save_model_dir = os.path.join("C:/Users/Robert/Desktop/College/Project/Model/Models/", NAME+".h5")
tensorboard = TensorBoard(log_dir = os.path.join("logs",NAME))
epoch = 15
seed_value = 123
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

#NOTE: X is a feature, y is a label
# load data - train
print('[+]: Loading pickled training data...')
X_train = pickle.load(open('../Data/ModelData/X_train.pickle', 'rb'))
y_train = pickle.load(open('../Data/ModelData/y_train.pickle', 'rb'))
# load data - test
print('[+]: Loading pickled test data...')
X_test = pickle.load(open('../Data/ModelData/X_test.pickle', 'rb'))
y_test = pickle.load(open('../Data/ModelData/y_test.pickle', 'rb'))
print('[+]: Pickled data loaded successfully...')

#reshape for RNN
X_train = np.reshape(X_train, (X_train.shape[0],  X_train.shape[1], 15))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 15))

def create_rnn():
    print("[+]: Creating model...")
    
    model = Sequential()
    model.add(LSTM(1000, input_shape=(128, 15)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', 
                    optimizer='adam', 
                    metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, 
                        epochs=epoch,
                        validation_data=(X_test, y_test),)
    #savemodel
    model.save(save_model_dir)

    ts_loss, ts_acc = model.evaluate(X_test,  y_test, verbose=2)    
    ts_loss = round(ts_loss,4)
    ts_acc = round(ts_acc,4)

    print("[+]: Model Generated!")
    print("[+]: Test Accuracy: "+ str(ts_acc))
    print("[+]: Test Loss: "+ str(ts_loss))
    print("[+]: Model - "+NAME)
    plot_complete_model(history,ts_acc)


def plot_complete_model(history,ts_acc):
    np.int(ts_acc)
    test_acc = "Test Accuracy:"+str(ts_acc)

    plt.title(NAME)
    plt.suptitle(test_acc+', Epochs:'+str(epoch))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.1, 1])
    plt.legend(loc='upper right')
    plt.savefig(fname = plot_name, dpi=100, bbox_inches=None, pad_inches=0)
    #plt.show()

create_rnn()
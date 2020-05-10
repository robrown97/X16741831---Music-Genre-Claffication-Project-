import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import time
from keras.utils import plot_model

#setup

#
#   naming convention ...
#   {network}-{num.hidden.layers}-{hl.1.size, hl.2.size, ...}-{drop.out}

NAME = "CNN-3-1024,512,1024-0.5-e.100-{}".format(int(time.time()))
plot_name = os.path.join("C:/Users/Robert/Desktop/College/Project/Model/Plots/", NAME+".png")
save_model_dir = os.path.join("C:/Users/Robert/Desktop/College/Project/Model/Models/", NAME+".h5")
tensorboard = TensorBoard(log_dir = os.path.join("logs",NAME))
epoch=100
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
#data transformation
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# CNN model
def create_cnn():
    print("[+]: Creating model...")
    model = Sequential()
    #input layer
    model.add(Conv2D(64,3,3, input_shape = (128,15,1), activation='relu') )
    model.add(Dropout(.5))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())

    #hidden layer
    model.add(Dense(1024))
    model.add(Activation("relu"))
    #hidden layer
    model.add(Dense(512))
    model.add(Activation("relu"))
    #hidden layer
    model.add(Dense(1024))
    model.add(Activation("relu"))

    #output layer
    model.add(Dense(10))
    model.add(Activation("softmax"))
    #compile
    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer="adam",
                    metrics=['accuracy'])
    model.summary()
    #run model
    history = model.fit(X_train, y_train, epochs = epoch, 
                        batch_size=32, 
                        validation_data=(X_test, y_test),
                        callbacks=[tensorboard])
    #savemodel
    model.save(save_model_dir)
   
    ts_loss, ts_acc = model.evaluate(X_test,  y_test, verbose=2)    
    print("[+]: Model Generated!")
    print("[+]: Test Accuracy: "+ str(ts_acc))
    print("[+]: Test Loss: "+ str(ts_loss))
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

create_cnn()


# IMDB
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
activations = [layers.core.Activation(keras.activations.sigmoid),
layers.core.Activation(keras.activations.relu), layers.LeakyReLU(),
layers.core.Activation(keras.activations.tanh)]
res=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
a=0

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1
    return results
m=0
for a in range(4):
    for b in range(4):
        losssum=0
        for c in range(6):
            print(f'{a} , {b} , {c}')
            x_train = vectorize_sequences(train_data)
            x_test = vectorize_sequences(test_data)
            y_train = np.asarray(train_labels).astype('float32')
            y_test = np.asarray(test_labels).astype('float32')
            x_val=x_train[:10000]
            partial_x_train=x_train[10000:]
            y_val=y_train[:10000]
            partial_y_train=y_train[10000:]
            from keras import models from keras import layers
            model=models.Sequential()
            model.add(layers.Dense(16, input_shape=(10000,)))
            model.add(activations[i])
            model.add(layers.Dense(1))
            model.add(activations[j])
            model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['acc'])
history=model.fit(partial_x_train, partial_y_train, epochs=5,batch_size=512, validation_data=(x_val,y_val))
losssum=losssum+history.history['val_loss'][len(history.history['val_loss'])-1]
print(res)

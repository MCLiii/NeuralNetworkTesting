
#MNIST

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist
activations=[keras.activations.sigmoid, keras.activations.relu,
keras.layers.LeakyReLU(), keras.activations.tanh]
results=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
class_names=[0,1,2,3,4,5,6,7,8,9]
a=0
for i in range(4):
    for j in range(4):
        losssum=0
            for k in range(6):
                (train_images, train_labels), (test_images, test_labels) = data.load_data()
                train_images = train_images/255.0
                test_images = test_images/255.0
                model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28,28)),
                keras.layers.Dense(128, activations[i]),
                keras.layers.Dense(10,  activations[j])
                # tanh softmax
                ])
                model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                history = model.fit(train_images, train_labels,
                validation_split=0.25, epochs=5, batch_size=16, verbose=1)
                prediction = model.predict(test_images)
                losssum=losssum+history.history['loss'][len(history.history['loss'])-1]
                results[a]=losssum/1
                a=a+1
 print(results)

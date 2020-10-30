import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = r"C:\Users\hp\Desktop\Career Excellence Maam\Neural Network\PetImage"

CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        break  # we just want one for now so break
    break  #...and one more!
print(img_array)

IMG_SIZE=100
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()


training_data = []
for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
    print(category)
    print(path)
    print(class_num)
    for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
        try:
            print(img)
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
        except OSError as e:
                print("OSErrroBad img most likely", e, os.path.join(path,img))
        except Exception as e:
                print("general exception", e, os.path.join(path,img))
print((training_data))

import random
random.shuffle(training_data)

print((training_data))


X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y).reshape(-1, 1)
print(len(y))
print(len(X))

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
print(X)
X = X/255.0
print(X)

NAME = "Cats-vs-dogs-CNN"
model = Sequential()

model.add(Conv2D(64, (3, 3), ))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('sigmoid'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )
print(X.shape)
model.fit(X, y,
          batch_size=32,
          epochs=25)

p=model.predict(X)
print(p)
print(y)

model.save("64x3-CNN.model")

for i in p:
     if i[0] < 0.5:
         print("dog")
     else:
         print("cat")
print("\n\n\n")
for i in y:
     if i[0] < 0.5:
         print("dog")
     else:
         print("cat")
model.save('64x3-CNN.model')
dir = r"C:\Users\hp\Desktop\Career Excellence Maam\Neural Network\SP"
  # create path to dogs and cats
for i in os.listdir(dir):  # iterate over each image per dogs and cats
    print(i)
    ia = cv2.imread(os.path.join(dir,i) ,cv2.IMREAD_GRAYSCALE)  # convert to array
    na = cv2.resize(ia, (100, 100))  # resize to normalize data size
    plt.imshow(na, cmap='gray')  # graph it
    plt.show()  # display!
  # we just want one for now so break
na=na/255.0
print(na)
print(na.shape)
na = np.array(na).reshape(-1, 100, 100, 1)
print(na.shape)
p=model.predict(na)

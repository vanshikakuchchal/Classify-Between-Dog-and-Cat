import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
model = tf.keras.models.load_model("64x3-CNN.model")
testdata=[]
dir = r"C:\Users\hp\Desktop\Career Excellence Maam\Neural Network\SP"
  # create path to dogs and cats
for i in os.listdir(dir):  # iterate over each image per dogs and cats
    print(i)
    ia = cv2.imread(os.path.join(dir,i) ,cv2.IMREAD_GRAYSCALE)  # convert to array
    na = cv2.resize(ia, (100, 100))  # resize to normalize data size
    plt.imshow(na, cmap='gray')  # graph it
    plt.show()  # display!
    na=na/255.0
    na = np.array(na).reshape(-1, 100, 100, 1)
    testdata.append(na)

for i in testdata:
    p=model.predict(i)
    if p< 0.5:
        print("dog")
    else:
        print("cat")

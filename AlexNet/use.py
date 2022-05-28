import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

AlexNet = tf.keras.model.load_model("AlexNet.h5")
CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
PATH_TO_IMAGE = ""

def process_image(image):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image
    



image = cv2.imread(PATH_TO_IMAGE) 
image = process_image(image)
image = tf.expand_dims(image, axis = 0) # has to be in batch form

predicted = AlexNet.predict(image)
label = CLASS_NAMES[np.argmax(predicted)]

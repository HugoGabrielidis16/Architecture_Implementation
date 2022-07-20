import matplotlib.pyplot as plt
from random import randint
import tensorflow as tf


def show_image(img, target):
    random_number = randint(0, len(img) - 6)
    plt.figure(figsize=(20, 10))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img[random_number + i + 1], "gray")
        plt.title("Original Image")
        plt.axis("off")
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        plt.imshow(target[random_number + i + 1], "gray")
        plt.title("Target Image")
        plt.axis("off")
    plt.show()


def rounded_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

import tensorflow as tf
import numpy as np


def encoder(inputs):
    x = tf.keras.layers.Conv2D(32, (2, 2), padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (2, 2), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(784)(x)
    return x


def decoder(x):
    x = tf.keras.layers.Dense(7 * 7 * 784)(x)
    x = tf.keras.layers.Reshape((7, 7, 784))(x)
    x = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2DTranspose(1, (2, 2), strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Activation("sigmoid")(x)
    return outputs


def create_model():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = encoder(inputs=inputs)
    print(x.shape)
    outputs = decoder(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1.5),
        loss="binary_crossentropy",
        metrics=["acc"],
    )

    return model


if __name__ == "__main__":
    model = create_model()
    x = tf.random.uniform((12, 28, 28, 1))
    y = model.predict(x)
    print(y.shape)

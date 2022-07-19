import tensorflow as tf


def encoder(inputs):
    x = tf.keras.layers.Conv2D(32, (2, 2), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    return x


def decoder(x):
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (2, 2), activation="relu", padding="same")(x)
    outputs = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    return outputs


def create_model():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1))
    x = encoder(inputs=inputs)
    outputs = decoder(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=["acc"])

    return model


if __name__ == "__main__":
    model = create_model()
    x = tf.random.uniform((1, 28, 28, 1))
    y = model.predict(x)
    print(y.shape)

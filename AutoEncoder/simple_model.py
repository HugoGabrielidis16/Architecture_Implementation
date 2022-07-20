from tensorflow import keras
from function import rounded_accuracy


def create_model():
    stacked_encoder = keras.models.Sequential(
        [
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(100, activation="selu"),
            keras.layers.Dense(30, activation="selu"),
        ]
    )
    stacked_decoder = keras.models.Sequential(
        [
            keras.layers.Dense(100, activation="selu", input_shape=[30]),
            keras.layers.Dense(28 * 28, activation="sigmoid"),
            keras.layers.Reshape([28, 28]),
        ]
    )
    stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
    stacked_ae.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.SGD(learning_rate=1.5),
        metrics=[rounded_accuracy],
    )
    return stacked_ae

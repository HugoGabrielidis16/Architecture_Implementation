import tensorflow as tf
from data import load_data
from model import create_model


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_data()
    model = create_model()

    model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        batch_size=32,
    )
    model.save("autoencoder.h5")
    # The encoder corresponds to the first 4 layers
    encoder = tf.keras.Sequential([])
    for layer in range(7):
        encoder.add(model.layers[layer])
    encoder.summary()

    for x, y in train_ds.take(1):
        y_pred = encoder.predict(x)
        print(y_pred.shape)

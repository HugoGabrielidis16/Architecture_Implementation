import tensorflow as tf
from data import load_data
from simple_model import create_model


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_data()
    model = create_model()
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            verbose=1,
        )
    ]
    model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        batch_size=32,
        callbacks=callbacks,
    )
    model.save("autoencoder.h5")

import tensorflow as tf
from sklearn.model_selection import train_test_split
from function import show_image


def add_dims(x, y):
    x = tf.expand_dims(x, axis=-1)
    y = tf.expand_dims(y, axis=-1)
    x = x / 255
    y = y / 255
    return x, y


def process_ds(ds, batch_size=32, augmented=False):
    """
    Takes as tf.data.Dataset map the processing image function to it, batch it to 32

    Parameters :
    ds (tf.data.Dataset) :
    """
    ds = ds.map(add_dims)
    ds = (
        ds.shuffle(buffer_size=1000)
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    return ds


def load_data():
    (X_train, _), (X_test, _) = tf.keras.datasets.fashion_mnist.load_data()

    X_train, X_val = train_test_split(
        X_train, test_size=0.1, random_state=1, shuffle=True
    )

    train_ds = process_ds(tf.data.Dataset.from_tensor_slices((X_train, X_train)))
    val_ds = process_ds(tf.data.Dataset.from_tensor_slices((X_val, X_val)))
    test_ds = process_ds(tf.data.Dataset.from_tensor_slices((X_test, X_test)))

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_data()
    for x, y in train_ds.take(1):
        print(x.shape)
        print(y.shape)
        show_image(x, y)

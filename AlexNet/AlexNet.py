import tensorflow as tf



(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

def process_images(image,label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image,label
    
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()

train_ds = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))
test_ds = (test_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))



model = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (227,227,3)),
    tf.keras.layers.Conv2D( filters = 96,
                            kernel_size = (11,11),
                            strides = 4,
                            activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (3,3),
                              strides = 2),
    tf.keras.layers.Conv2D( filters = 256,
                            kernel_size = (5,5), 
                            padding = 'same',
                            activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (3,3),
                              strides = 2),
    tf.keras.layers.Conv2D( filters = 384,
                            kernel_size = (3,3),
                            strides = 1,
                            padding = 'same'),
    tf.keras.layers.Conv2D( filters = 384,
                            kernel_size = (3,3),
                            strides = 1,
                            padding = 'same'),
    tf.keras.layers.Conv2D( filters = 256,
                            kernel_size = (3,3),
                            strides = 1,
                            padding = 'same'),
    tf.keras.layers.MaxPool2D(pool_size = (3,3),
                              strides = 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 4096, activation = 'relu'),
    tf.keras.layers.Dropout( rate = 0.5),
    tf.keras.layers.Dense(units = 4096, activation = 'relu'),
    tf.keras.layers.Dropout( rate = 0.5),
    tf.keras.layers.Dense(10, activation = 'softmax')

])

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)





model.fit( train_ds,
           epochs = 2,
           validation_data = test_ds,
           validation_batch_size = 32)

model.save("AlexNet.h5")



CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
import tensorflow as tf


class IdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name="")

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.Activation("relu")
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, input_tensor])
        x = self.act(x)

        return x


class ResNet18(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.conv = tf.keras.layers.Conv2D(64, 7, padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation("relu")
        self.max_pool = tf.keras.layers.MaxPool2D(3)
        self.id1a = IdentityBlock(64, 3)
        self.id1b = IdentityBlock(64, 3)
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)
        x = self.id1a(x)
        x = self.id1b(x)

        x = self.global_pool(x)

        return self.classifier(x)


resnet18 = ResNet18(10)

resnet18.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)


(train_images, train_labels), (
    test_images,
    test_labels,
) = tf.keras.datasets.cifar10.load_data()


def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()

train_ds = (
    train_ds.map(process_images)
    .shuffle(buffer_size=train_ds_size)
    .batch(batch_size=32, drop_remainder=True)
)
test_ds = (
    test_ds.map(process_images)
    .shuffle(buffer_size=train_ds_size)
    .batch(batch_size=32, drop_remainder=True)
)


resnet18.fit(train_ds, epochs=2, validation_data=test_ds, validation_batch_size=32)

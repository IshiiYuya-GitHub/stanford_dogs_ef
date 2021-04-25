import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import datetime
import os


IMG_SIZE = 224
batch_size = 64

dataset_name = "stanford_dogs"
(ds_train, ds_val, ds_test), ds_info = tfds.load(dataset_name, split=[
    "train", "test[:50%]", "test[50%:100%]"], with_info=True, as_supervised=True)

NUM_CLASSES = ds_info.features["label"].num_classes

size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (
    tf.image.resize(image, size), label))
ds_val = ds_val.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (
    tf.image.resize(image, size), label))
img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x,
                           weights="efficientnetb0_notop.h5")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_val = ds_val.map(
    input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_val = ds_val.batch(batch_size=batch_size, drop_remainder=True)

ds_test = ds_test.map(
    input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

model = build_model(num_classes=NUM_CLASSES)

# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir=log_dir, histogram_freq=1)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# チェックポイントコールバックを作る
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=2)

model.load_weights(checkpoint_path)

# epochs = 25
# hist = model.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=2)

# # history = model.fit(ds_train,
# #                     epochs=epochs,
# #                     validation_data=ds_val,
# #                     callbacks=[tensorboard_callback])

# unfreeze_model(model)
# fine_epochs = 10  # @param {type: "slider", min:8, max:50}
# hist = model.fit(ds_train, initial_epoch=epochs,
#                  epochs=epochs+fine_epochs, validation_data=ds_val, verbose=2, callbacks=[cp_callback])

loss, accuracy = model.evaluate(ds_test)
print('Test accuracy :', accuracy)

# Retrieve a batch of images from the test set
image_batch, label_batch = ds_test.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

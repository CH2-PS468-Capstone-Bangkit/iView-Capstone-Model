# # Import semua library yang dibutuhkan
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.activations import relu, softmax
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

# # Import dataset
train_directory = 'images/train'
validation_directory = 'images/validation'

# # Buat datagen
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
validation_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest',
)

# # Buat generator
train_generator = train_datagen.flow_from_directory(
    directory=train_directory,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='sparse',
    # batch_size=32,
)
validation_generator = validation_datagen.flow_from_directory(
    directory=validation_directory,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='sparse',
    # batch_size=32,
)

# # Buat model
model = Sequential([
    InputLayer(input_shape=(48, 48, 1)),
    Conv2D(filters=128, kernel_size=(3, 3), activation=relu),
    MaxPool2D(),
    Conv2D(filters=256, kernel_size=(3, 3), activation=relu),
    MaxPool2D(),
    Conv2D(filters=512, kernel_size=(3, 3), activation=relu),
    MaxPool2D(),
    Flatten(),
    Dense(units=1024, activation=relu),
    Dropout(rate=0.25),
    Dense(units=512, activation=relu),
    Dropout(rate=0.25),
    Dense(units=256, activation=relu),
    Dropout(rate=0.25),
    Dense(units=7, activation=softmax),
])
model.summary()

# # Deklarasikan callback
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')>0.8 and logs.get('val_accuracy')>0.8:
            print("\nTraining dihentikan karena 'accuracy' dan 'val_accuracy' telah mencapai 80%")
            self.model.stop_training = True
callbacks = MyCallback()

# # Coba model
model.compile(
    optimizer=Adam(learning_rate=0.01), 
    loss=SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy'],
)
model.fit(
    x=train_generator, 
    epochs=10, 
    steps_per_epoch=100,
    validation_data=(validation_generator), 
    callbacks=callbacks,
)

# # Simpan model
# CC
model.save("result_model.h5")

# MD
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('result_model.tflite', 'wb') as f:
    f.write(tflite_model)

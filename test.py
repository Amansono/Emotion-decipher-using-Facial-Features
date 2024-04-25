import tensorflow as tf
import numpy as numpy
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.layers import Activation, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

train_dir = r'C:\Users\youbo\Desktop\ML TERM PROJECT\archive\train'
test = r'C:\Users\youbo\Desktop\ML TERM PROJECT\archive\test'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1.0/255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255)

 
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode = "grayscale",
        shuffle=True,
        class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
        test,
        target_size=(48,48),
        batch_size=64,
        color_mode = "grayscale",
        shuffle=True,
        class_mode='categorical')

model= Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128,(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('best.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

callbacks = [checkpoint]

opt = tf.keras.optimizers.Adam(lr=0.0005)

model.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])

history = model.fit_generator(train_generator,epochs=100,validation_data=validation_generator,callbacks=callbacks)
model.save('best.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

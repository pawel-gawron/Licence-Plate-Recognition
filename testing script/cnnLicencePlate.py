# IMport Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
import sys

import tensorflow as tf

from matplotlib.image import imread
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

import warnings
warnings.filterwarnings('ignore')

localization = os.path.abspath(os.path.dirname(__file__))
master_catalog = os.path.abspath(os.path.join(localization, '..'))

dir_path = 'dataset/ocrDataset/data'
model_path = 'model/model.tflite'

save_model_path = os.path.join(master_catalog, model_path)

dir_path = os.path.join(master_catalog, dir_path)


digits = sorted(os.listdir(dir_path))
NUM_CLASSES = len(digits)
print(digits)
print('Number of classes (letters and digits): ', NUM_CLASSES)

digits_counter = {}
NUM_IMAGES = 0

for digit in digits:
    path = os.path.join(dir_path, digit)
    digits_counter[digit] = len(os.listdir(path))
    NUM_IMAGES += len(os.listdir(path))

print(digits_counter)
print('Number of all images: ', NUM_IMAGES)

data = []
labels = []
MAX_NUM = None   # maximum number of digits images per class
IMG_WIDTH, IMG_HEIGHT = 32, 40

for digit in digits:
    path = os.path.join(dir_path, digit)
    label = digits.index(digit)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_array = cv2.imread(img_path)
        resized = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)   
        # ret3, gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)     
        data.append(gray)
        labels.append(label)
        if MAX_NUM is not None:
            if labels.count(label) == MAX_NUM:
                break

data = np.array(data, dtype='float32')
labels = np.array(labels, dtype='int8')

print("labels: ", labels)

data = data / 255.0
data = data.reshape(*data.shape, 1)
# labels = to_categorical(labels, num_classes=NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(data, labels, shuffle=True, test_size=.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, test_size=.3, random_state=42)

print("Training dataset shape: ", X_train.shape, y_train.shape)
print("Validation dataset shape: ", X_val.shape, y_val.shape)
print("Testing dataset shape: ", X_test.shape, y_test.shape)

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model.summary()

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

checkpoint_path = "content/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(X_train, y_train, epochs=20, batch_size=256,
                    validation_data=(X_val, y_val),
                    callbacks=[cp_callback, es_callback])

# Zapisz model w formacie TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Ustaw opcje kwantyzacji
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Zapisz model do pliku
with open(save_model_path, 'wb') as f:
    f.write(tflite_model)

tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)

# Save the entire model as a SavedModel.
model.save('content/saved_model/my_model')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print(test_acc)
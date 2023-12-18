#!/usr/bin/env python
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import decode_predictions
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

input_folder = './data'

zip_file_path = input_folder + '/vegetables.zip'
extract_dir = input_folder

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the specified directory
    zip_ref.extractall(extract_dir)

print("Extraction complete!")

img = load_img(input_folder + '/Vegetable Images/train/Carrot/0001.jpg', target_size=(224, 224))
img
x = np.array(img)

model = EfficientNetB0(weights='imagenet', input_shape=(224, 224, 3))
X = np.array([x])
X = preprocess_input(X)
pred = model.predict(X)
decode_predictions(pred)

main_directory = input_folder + '/Vegetable Images'
train_dir = os.path.join(main_directory, 'train')
validation_dir = os.path.join(main_directory, 'validation')
test_dir = os.path.join(main_directory, 'test')

# # Transfer learning with EfficientNetB0

train_data = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),  # EfficientNetB0 input size
    batch_size=32,
    shuffle=True,
    seed=42
)

validation_data = image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),  # EfficientNetB0 input size
    batch_size=32,
    shuffle=False
)

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the base model layers
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(15, activation='softmax')  # Adjust 15 for your number of classes
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data,
                    validation_data=validation_data,
                    epochs=10)  # You can adjust the number of epochs

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()


# # Learning rate adjusting

def trainModel(learning_rate):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(15, activation='softmax')  # Adjust 15 for your number of classes
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

lr_scores = {}

for lr in [0.0001, 0.001, 0.01, 0.1]:
    print(lr)
    
    model = trainModel(lr)
    history = model.fit(train_data,
                    validation_data=validation_data,
                    epochs=10)
    lr_scores[lr] = history.history
    print()
    print()

for lr, hist in lr_scores.items():
    plt.plot(hist['val_accuracy'], label=lr)
    plt.xticks(np.arange(10))
    plt.legend()

del lr_scores[0.1]
for lr, hist in lr_scores.items():
    plt.plot(hist['val_accuracy'], label=lr)
    plt.xticks(np.arange(10))
    plt.legend()

# # Customize dense layer

def trainDense(denseSize):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(denseSize, activation='relu'),
    layers.Dense(15, activation='softmax')  # Adjust 15 for your number of classes
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

dense_scores = {}

for ds in [128, 256, 512, 1024]:
    print(ds)
    
    model = trainDense(ds)
    history = model.fit(train_data,
                    validation_data=validation_data,
                    epochs=10)
    dense_scores[ds] = history.history
    print()
    print()

for ds, hist in dense_scores.items():
    plt.plot(hist['val_accuracy'], label=ds)
    plt.xticks(np.arange(10))
    plt.legend()

# # Regularization and dropout

def trainDropout(dropout_rate):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(dropout_rate),
    layers.Dense(15, activation='softmax')  # Adjust 15 for your number of classes
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

dp_scores = {}

for dp in [0.0, 0.2, 0.5, 0.8]:
    print(dp)
    
    model = trainDropout(dp)
    history = model.fit(train_data,
                    validation_data=validation_data,
                    epochs=10)
    dp_scores[dp] = history.history
    print()
    print()

for dp, hist in dp_scores.items():
    plt.plot(hist['val_accuracy'], label=dp)
    plt.xticks(np.arange(10))
    plt.legend()

# # Final model

def finalModel(learning_rate, dropout_rate, dense_rate):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(dense_rate, activation='relu'),
    layers.Dropout(dropout_rate),
    layers.Dense(15, activation='softmax')  # Adjust 15 for your number of classes
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

checkpoint = keras.callbacks.ModelCheckpoint(
             'efficinentB0_best.h5',
             save_best_only=True,
             monitor='val_accuracy',
             mode='max')

learning_rate = 0.001
dropout_rate = 0.5
dense_rate = 128

model = finalModel(learning_rate, dropout_rate, dense_rate)
history = model.fit(train_data,
                    validation_data=validation_data,
                    epochs=10,
                    callbacks=[checkpoint])


# # Using the model

test_data = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(224, 224),  # EfficientNetB0 input size
    batch_size=32,
    shuffle=False
)

model = keras.models.load_model('efficinentB0_best.h5')
model.evaluate(test_data)

papaya_path = input_folder + '/Vegetable Images/test/Papaya/1198.jpg'
papaya = load_img(papaya_path, target_size = (224, 224))
x = np.array(papaya)
X = np.array([x])
X = preprocess_input(X)
pred = model.predict(X)
pred
class_labels = test_data.class_names
print("Class Labels:", class_labels)

preds = dict(zip(class_labels, pred[0]))
max_value = max(preds[label] for label in class_labels)
max_label = [label for label in class_labels if preds[label] == max_value]

print(f"The maximum value among the class labels is: {max_value}")
print(f"The label(s) associated with the maximum value is/are: {max_label}")

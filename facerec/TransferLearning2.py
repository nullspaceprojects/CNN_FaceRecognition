#import os
#import pandas as pd
#import numpy as np
#import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
#from keras_vggface import utils
import datetime
import pickle
#import tensorflow as tf

#============ LOAD THE DATASET ======================
train_dir = r'./facedb/train'
validation_dir = r'./facedb/validation'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
FACE_LABEL_FILENAME = './labels/face-labels.pickle'

#TRAIN DATASET
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=IMG_SIZE,color_mode='rgb',
                                                    batch_size=BATCH_SIZE,class_mode='categorical',shuffle=True)
print(train_generator.class_indices.values())
# dict_values([0, 1, 2])
NO_CLASSES = len(train_generator.class_indices.values())
print(NO_CLASSES)
#VALIDATION DATASET
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = train_datagen.flow_from_directory(validation_dir,target_size=IMG_SIZE,color_mode='rgb',
                                                    batch_size=BATCH_SIZE,class_mode='categorical',shuffle=True)
print(validation_generator.class_indices.values())

#============================ LOAD THE BASE MODEL ====================
#include_top=False -> discharge the last fully connected layers
# it will be removed:
# avg_pool (GlobalAveragePooling  (None, 2048)        0           ['conv5_block3_out[0][0]']
# predictions (Dense)            (None, 1000)         2049000     ['avg_pool[0][0]']
# True->177 layers
# False->175 layers
IMG_SHAPE = IMG_SIZE + (3,)
#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
#'senet50' Based on SENET50 architecture -> new paper(2017)
#Downloading data from https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_senet50.h5
MODEL_TYPE = 'vgg16'
base_model = VGGFace(model=MODEL_TYPE, include_top=False, input_shape=IMG_SHAPE)
#freeze all network ResNet50 layers
base_model.trainable = False
'''
for layer in base_model.layers:
    layer.trainable = False
'''
base_model.summary()
print(len(base_model.layers))
print(base_model.output)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
# final layer with softmax activation
predictions = Dense(NO_CLASSES, activation='softmax')(x)
#predictions = Dense(1)(x)
print(predictions)
# this is the model we will train
model = Model(inputs=base_model.inputs, outputs=predictions)
# don't train the first 19 layers - 0..18
for layer in model.layers[:19]:
    layer.trainable = False

# train the rest of the layers - 19 onwards
for layer in model.layers[19:]:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
'''
base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
'''
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.summary()
#TRAIN THE MODEL
initial_epochs = 5
loss0, accuracy0 = model.evaluate(validation_generator)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(train_generator,
                    epochs=initial_epochs,
                    validation_data=validation_generator)

# creates a HDF5 file
dtnow = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
model.save(f'./saved_models/{MODEL_TYPE}_{dtnow}_cnn.h5')



class_dictionary = train_generator.class_indices
class_dictionary = {value: key for key, value in class_dictionary.items()}
print(class_dictionary)
# save the class dictionary to pickle
with open(FACE_LABEL_FILENAME, 'wb') as f:
    pickle.dump(class_dictionary, f)

#LEARNING CURVES
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

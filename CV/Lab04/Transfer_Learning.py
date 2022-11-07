# coding: utf-8

# # Transfer Learning
# 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import applications
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras import models
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from Lab04.util import create_image, predict_image, load_model_and_predict, crop, contrast, brightness

if __name__ == '__main__':
    print('TensorFlow version:', tf.__version__)
    print('Keras version:', keras.__version__)

    # ## Prepare the base model
    base_model = keras.applications.resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print(base_model.summary())

    data_folder = 'data'
    pretrained_size = (224, 224)
    batch_size = 30

    print("Getting Data...")
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 # horizontal_flip=True,
                                 # width_shift_range=[-20, 20],
                                 # rotation_range=90,
                                 # brightness_range=[0.2, 1.0],
                                 validation_split=0.3)

    print("Preparing training dataset...")
    train_generator = datagen.flow_from_directory(
        data_folder,
        target_size=pretrained_size,  # resize to match model expected input
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')  # set as training data

    print("Preparing validation dataset...")
    validation_generator = datagen.flow_from_directory(
        data_folder,
        target_size=pretrained_size,  # resize to match model expected input
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')  # set as validation data

    classnames = list(train_generator.class_indices.keys())
    print("class names: ", classnames)

    # ## Create a prediction layer
    #

    # Freeze the already-trained layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Create prediction layer for classification of our images
    x = base_model.output
    x = Flatten()(x)
    prediction_layer = Dense(len(classnames), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction_layer)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Now print the full model
    print(model.summary())

    # ## Train the Model
    #
    checkpoint_filepath = 'models/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    num_epochs = 3
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[model_checkpoint_callback],
        epochs=num_epochs)

    # ## View the loss history
    epoch_nums = range(1, num_epochs + 1)
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]
    plt.plot(epoch_nums, training_loss)
    plt.plot(epoch_nums, validation_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

    # ## Evaluate model performance
    #

    print("Generating predictions from validation data...")
    # Get the image and label arrays for the first batch of validation data
    x_test = validation_generator[0][0]
    y_test = validation_generator[0][1]

    # Use the model to predict the class
    class_probabilities = model.predict(x_test)

    # The model returns a probability value for each class
    # The one with the highest probability is the predicted class
    predictions = np.argmax(class_probabilities, axis=1)

    # The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1
    true_labels = np.argmax(y_test, axis=1)

    # Plot the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classnames))
    plt.xticks(tick_marks, classnames, rotation=85)
    plt.yticks(tick_marks, classnames)
    plt.xlabel("Predicted Shape")
    plt.ylabel("Actual Shape")
    plt.show()

    # Create a random test image
    classnames = os.listdir('data')
    classnames.sort()
    rand_shape = classnames[randint(0, len(classnames) - 1)]
    img = create_image((224, 224), rand_shape)

    augmentations = [crop, contrast, brightness]
    for fns in augmentations:
        _img = fns(img)
        plt.imshow(_img)
        plt.show()

        # Use the classifier to predict the class
        class_idx = predict_image(model, img)
        print(f'1: got {classnames[class_idx]}, expected {rand_shape} ({fns.__name__})')

        # Use the classifier to predict the class
        class_idx = predict_image(model, _img)
        print(f'2: got {classnames[class_idx]}, expected {rand_shape} ({fns.__name__})')

        # Use best model
        class_idx = load_model_and_predict(model, checkpoint_filepath, _img)
        print(f'3: got {classnames[class_idx]}, expected {rand_shape} ({fns.__name__})')

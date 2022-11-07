# Function to predict the class of an image
from random import randint
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf


def predict_image(classifier, image):
    from tensorflow import convert_to_tensor
    # The model expects a batch of images as input, so we'll create an array of 1 image
    imgfeatures = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    imgfeatures = imgfeatures.astype('float32')
    imgfeatures /= 255

    # Use the model to predict the image class
    class_probabilities = classifier.predict(imgfeatures)

    # Find the class predictions with the highest predicted probability
    index = int(np.argmax(class_probabilities, axis=1)[0])
    return index


# Function to create a random image (of a square, circle, or triangle)
def create_image(size, shape):
    xy1 = randint(10, 40)
    xy2 = randint(60, 100)
    col = (randint(0, 200), randint(0, 200), randint(0, 200))

    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if shape == 'circle':
        draw.ellipse([(xy1, xy1), (xy2, xy2)], fill=col)
    elif shape == 'triangle':
        draw.polygon([(xy1, xy1), (xy2, xy2), (xy2, xy1)], fill=col)
    else:  # square
        draw.rectangle([(xy1, xy1), (xy2, xy2)], fill=col)
    del draw

    return np.array(img)


def rotate(func):
    def func_wrapper(*args, **kwargs):
        if randint(0, 1) == 0:
            return func(*args, **kwargs)
        return tf.image.rot90(func(*args, **kwargs)).numpy()
    return func_wrapper


def load_model_and_predict(model, path, image):
    model.load_weights(path)
    return predict_image(model, image)


@rotate
def brightness(image):
    seed = (randint(0, 3), 0)
    return tf.image.stateless_random_brightness(
        image, max_delta=0.95, seed=seed).numpy()


@rotate
def contrast(image):
    seed = (randint(0, 3), 0)
    return tf.image.stateless_random_contrast(
        image, lower=0.1, upper=0.9, seed=seed).numpy()


@rotate
def crop(image):
    seed = (randint(0, 3), 0)
    return tf.image.stateless_random_crop(
      image, size=[224, 224, 3], seed=seed).numpy()

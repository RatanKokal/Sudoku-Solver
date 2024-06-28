import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(main_directory = './assets/training_images/', batch_size = 8, validation_split = 0.1):

    image_size = (28, 28)  # Image size
    num_classes = 10  # Number of classes

    # Create TensorFlow training datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        './assets/training_images/',
        labels = 'inferred',
        label_mode = 'categorical',
        color_mode = 'grayscale',
        image_size = image_size,
        batch_size = batch_size,
        shuffle = True,
        seed = 42,
        validation_split = validation_split,
        subset = 'training'
    )

    # Create TensorFlow validation datasets
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        './assets/training_images/',
        labels = 'inferred',
        label_mode = 'categorical',
        color_mode = 'grayscale',
        image_size = image_size,
        batch_size = batch_size,
        shuffle = True,
        seed = 42,
        validation_split = validation_split,
        subset = 'validation'
    )

    preprocessed_data = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),  # Normalize images
    ])

    # Apply the rescaling layer to your dataset
    train_ds = train_ds.map(lambda x, y: (preprocessed_data(x), y))
    validation_ds = validation_ds.map(lambda x, y: (preprocessed_data(x), y))

    return train_ds, validation_ds



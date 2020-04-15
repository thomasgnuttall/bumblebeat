import tensorflow as tf
import tensorflow_datasets as tfds

def load_midi_to_tf(path):
    """
    Load midi to tensorflow dataset

    :param path: path to midi files
    :type path: str

    :return: tensorflow dataset
    :rtype: tensorflow.data.Dataset
    """
    
    dataset = tfds.load(
        name=path, 
        split=tfds.Split.TRAIN,
        try_gcs=True)

    # Build your input pipeline
    dataset = dataset.shuffle(1024).batch(32).prefetch(
        tf.data.experimental.AUTOTUNE)

    for features in dataset.take(1):
        # Access the features you are interested in
        midi, genre = features["midi"], features["style"]["primary"]

    return dataset
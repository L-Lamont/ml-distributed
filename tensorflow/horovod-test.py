import os
import logging
import math

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils

import tensorflow as tf
from tensorflow.keras import layers
import horovod.tensorflow as hvd
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('--num-epochs', type=int, help='Set the number of epochs', default=12)
    add_arg('--batch-size', type=int, help='Set the global batch size', default=32)
    add_arg('--output', help='Specify the output directory')
    add_arg('--data', help='Specify the input data location')

    return parser.parse_args()


def config_logging(filename):
    """Configures logging"""
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=log_format, filename=filename)


def decay(epoch):
    """Defines learning rate at different epochs"""
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


def main():
    """Main function"""

    args = parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    output = args.output
    data = args.data

    logfile = os.path.join(output, 'logfile')
    tensorboard_dir = os.path.join(output, 'tensorboard')

    config_logging(filename=logfile)

    # Call to horovod init
    hvd.init()

    # Set 1 GPU visible per process
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path=data)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=data)

    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    # Wrap the optimizer in hvd.DistributedOptimizer
    model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = hvd.DistributedOptimizer(tf.keras.optimizers.Adam()),
        metrics = ['accuracy']
    )

    # Broadcast variables
    if(hvd.rank() == 0):
        hvd.broadcast_variables

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir),
        tf.keras.callbacks.LearningRateScheduler(decay)
    ]

    model.fit(
        x=train_images,
        y=train_labels,
        epochs=num_epochs,
        batch_size=batch_size,
        shuffle=True
    )

    eval_loss, eval_acc = model.evaluate(x=test_images, y=test_labels)


if __name__ == '__main__':
    main()

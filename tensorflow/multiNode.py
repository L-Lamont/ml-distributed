import os
import logging
import atexit

import argparse
import tensorflow as tf
from tensorflow.keras import layers

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('--num-epochs', type=int, help='Set the number of epochs', default=12)
    add_arg('--batch-size', type=int, help='Set the global batch size', default=32)
    add_arg('--output', help='Set the output directory')
    add_arg('--data', help='Specify the directory with input')

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

    # Allow dynamic memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Specify the distribution strategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=data)

    # Convert dataset to a tensorflow dataset
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    # Update the batchsize
    train_data = train_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    # Update the sharding policy (Because we are using such a single file dataset)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_data = train_data.with_options(options)
    test_data = test_data.with_options(options)

    # Wrap the Model definition and compilation
    with strategy.scope():
        model = tf.keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])

        model.compile(
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir),
        tf.keras.callbacks.LearningRateScheduler(decay),
    ]

    model.fit(
        train_data,
        epochs=num_epochs, 
        shuffle=True,
        callbacks=callbacks
    ) 

    eval_loss, eval_acc = model.evaluate(test_data)

    print()
    print('Eval_loss\t {}\nEval_accuracy\t {}'.format(eval_loss, eval_acc))
    print()


if __name__ == '__main__':
    main()

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

    print('Test')
    print(output)
    print(data)
    

    logfile = os.path.join(output, 'logfile')
    tensorboard_dir = os.path.join(output, 'tensorboard')

    config_logging(filename=logfile)
    
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=data)


    # Specify the distribution strategy
    strategy = tf.distribute.MirroredStrategy()

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
        x=train_images, 
        y=train_labels, 
        epochs=num_epochs, 
        batch_size=batch_size,
        shuffle=True
    ) 

    eval_loss, eval_acc = model.evaluate(x=test_images, y=test_labels)

    # Close the threadpool created and not closed by tf
    atexit.register(strategy._extended._collective_ops._pool.close)


if __name__ == '__main__':
    main()

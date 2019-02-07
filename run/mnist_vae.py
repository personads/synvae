import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from models.visual import MnistVae

from experiments import *


if __name__ == '__main__':
    args = parse_arguments('MNIST VAE').parse_args()
    model_path, tb_path, out_path, log_path = make_experiment_dir(args.exp_path)
    setup_logging(log_path)
    # set up model
    mnist_vae = MnistVae(latent_dim=50, batch_size=args.batch_size)
    # load MNIST
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    logging.info("Loaded %d training, %d test images from MNIST." % (train_images.shape[0], test_images.shape[0]))
    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.
    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.
    # set up TF datasets
    num_batches = train_images.shape[0] // args.batch_size
    num_batches_test = test_images.shape[0] // args.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(args.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    train_next = train_iterator.get_next()
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(args.batch_size)
    test_iterator = test_dataset.make_initializable_iterator()
    test_next = test_iterator.get_next()

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize FileWriter for TensorBoard
        tf_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)
        # initialize variables
        sess.run(tf.global_variables_initializer())
        mnist_vae.train(sess, train_iterator, test_iterator, epochs, model_path, out_path, tf_writer)
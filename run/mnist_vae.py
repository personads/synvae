import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from models.visual import MnistVae

from utils.experiments import *


if __name__ == '__main__':
    args = parse_arguments('MNIST VAE').parse_args()
    model_path, tb_path, out_path, log_path = make_experiment_dir(args.exp_path)
    setup_logging(log_path)
    # set up model
    mnist_vae = MnistVae(latent_dim=50, beta=args.beta, batch_size=args.batch_size)
    mnist_vae.build()
    # load MNIST
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    valid_images = train_images[50000:]
    train_images = train_images[:50000]
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    valid_images = valid_images.reshape(valid_images.shape[0], 28, 28, 1).astype('float32')
    logging.info("Loaded %d training, %d validation images from MNIST." % (train_images.shape[0], valid_images.shape[0]))
    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    valid_images /= 255.
    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    valid_images[valid_images >= .5] = 1.
    valid_images[valid_images < .5] = 0.
    # set up TF datasets
    num_batches = train_images.shape[0] // args.batch_size
    num_batches_test = valid_images.shape[0] // args.batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(args.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_images).batch(args.batch_size)
    valid_iterator = valid_dataset.make_initializable_iterator()

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # initialize FileWriter for TensorBoard
        tf_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)
        # training loop
        mnist_vae.train(sess, train_iterator, valid_iterator, epochs, model_path, out_path, tf_writer)

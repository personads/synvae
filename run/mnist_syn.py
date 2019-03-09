import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from models.visual import MnistVae
from models.auditive import MusicVae
from models.synesthetic import SynestheticVae

from experiments import *


if __name__ == '__main__':
    arg_parser = parse_arguments('SynVAE (MNIST)')
    arg_parser.add_argument('musicvae_config', help='name of the MusicVAE model configuration (e.g. hierdec-mel_16bar)')
    arg_parser.add_argument('musicvae_path', help='path to MusicVAE model checkpoints')
    args = arg_parser.parse_args()
    model_path, tb_path, out_path, log_path = make_experiment_dir(args.exp_path)
    setup_logging(log_path)

    # set up auditive model
    music_vae = MusicVae(config_name=args.musicvae_config, batch_size=args.batch_size)
    # set up visual model
    mnist_vae = MnistVae(latent_dim=music_vae.latent_dim, batch_size=args.batch_size)
    # set up synesthetic model
    model = SynestheticVae(visual_model=mnist_vae, auditive_model=music_vae, learning_rate=1e-4)
    model.build()

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
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(args.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    train_next = train_iterator.get_next()
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(args.batch_size)
    test_iterator = test_dataset.make_initializable_iterator()
    test_next = test_iterator.get_next()

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # restore MusicVAE
        model.restore_auditive(tf_session=sess, path=args.musicvae_path)
        # set up TensorBoard writer
        tf_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)
        # training loop
        model.train(sess, train_iterator, test_iterator, epochs, model_path, out_path, tf_writer)
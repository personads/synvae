import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from data.cifar import Cifar
from models.visual import CifarVae

from experiments import *


if __name__ == '__main__':
    arg_parser = parse_arguments('CIFAR10 VAE')
    arg_parser.add_argument('cifar_path', help='path to CIFAR10')
    args = arg_parser.parse_args()
    model_path, tb_path, out_path, log_path = make_experiment_dir(args.exp_path)
    setup_logging(log_path)
    # set up model
    cifar_vae = CifarVae(latent_dim=512, batch_size=args.batch_size)
    # load CIFAR
    cifar = Cifar(args.cifar_path)
    train_images = cifar.data[:int(cifar.data.shape[0]*.8)]
    valid_images = cifar.data[int(cifar.data.shape[0]*.8):]
    logging.info("Loaded %d training, %d validation images from CIFAR10." % (train_images.shape[0], valid_images.shape[0]))
    # set up TF datasets
    num_batches = train_images.shape[0] // args.batch_size + 1
    num_batches_valid = valid_images.shape[0] // args.batch_size + 1
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(args.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_images).batch(args.batch_size)
    valid_iterator = valid_dataset.make_initializable_iterator()
    logging.info("Loaded data into TensorFlow.")

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize FileWriter for TensorBoard
        tf_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)
        # initialize variables
        sess.run(tf.global_variables_initializer())
        cifar_vae.train(sess, train_iterator, valid_iterator, epochs, model_path, out_path, tf_writer)
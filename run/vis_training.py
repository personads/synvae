import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from models.visual import *

from utils.experiments import *


if __name__ == '__main__':
    arg_parser = parse_arguments('VisualVAE - Training')
    args = arg_parser.parse_args()
    model_path, tb_path, out_path, log_path = make_experiment_dir(args.exp_path)
    setup_logging(log_path)

    # set up visual model
    if args.task == 'mnist':
        model = MnistVae(latent_dim=50, beta=args.beta, batch_size=args.batch_size)
    elif args.task == 'cifar':
        model = CifarVae(latent_dim=512, beta=args.beta, batch_size=args.batch_size)
    model.build()

    # load data
    images, labels, label_descs, num_labels = load_data(args.task, split='train', data_path=args.data_path)
    train_images, _, valid_images, _ = split_data(images, labels, args.task)

    # set up TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(args.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_images).batch(args.batch_size)
    valid_iterator = valid_dataset.make_initializable_iterator()

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # set up TensorBoard writer
        tf_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)
        # training loop
        model.train(sess, train_iterator, valid_iterator, epochs, model_path, out_path, tf_writer)
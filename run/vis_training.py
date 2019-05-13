import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from data import *
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
        dataset = Mnist(split='train', data_path=args.data_path)
    elif args.task == 'cifar':
        model = CifarVae(latent_dim=512, beta=args.beta, batch_size=args.batch_size)
        dataset = Cifar(args.data_path)
    elif args.task == 'bam':
        model = BamVae(latent_dim=512, beta=args.beta, batch_size=args.batch_size)
        dataset = Bam(args.data_path)
    model.build()

    # load data
    train_iterator, valid_iterator = dataset.get_train_image_iterators(batch_size=args.batch_size)

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # training loop
        model.train(sess, train_iterator, valid_iterator, epochs, model_path, out_path)

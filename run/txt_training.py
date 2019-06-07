import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from data import *
from models.textual import *

from utils.experiments import *


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='TextualVAE - Training')
    arg_parser.add_argument('task', choices=['sarc'], help='name of the task')
    arg_parser.add_argument('exp_path', help='path to experiment files (model checkpoints, TensorBoard logs, model outputs)')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('--beta', type=float, default=1., help='beta parameter for weighting KL-divergence (default: 1.0)')
    arg_parser.add_argument('--epochs', type=int, default=100, help='number of training epochs (default: 100)')
    arg_parser.add_argument('--init_epoch', type=int, default=0, help='epoch to resume at (default: 0)')
    arg_parser.add_argument('--batch_size', type=int, default=32, help='batch size for training and evaluation (default: 200)')
    args = arg_parser.parse_args()
    model_path, tb_path, out_path, log_path = make_experiment_dir(args.exp_path)
    setup_logging(log_path)

    # set up model
    if args.task == 'sarc':
        dataset = Sarc(data_path=args.data_path)
        model = SarcVae(idx_tkn_map=dataset.idx_tkn_map, max_length=dataset.max_length, latent_dim=512, beta=args.beta, batch_size=args.batch_size)
    model.build()

    # load data
    train_iterator, valid_iterator = dataset.get_train_image_iterators(batch_size=args.batch_size) # "image" -> texts

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # restore model when resuming training
        if args.init_epoch > 0:
            model.restore(tf_session=sess, path=os.path.join(model_path, 'latest_model.ckpt'))
            model.epoch = args.init_epoch - 1
        # training loop
        model.train(sess, train_iterator, valid_iterator, epochs, model_path, out_path)

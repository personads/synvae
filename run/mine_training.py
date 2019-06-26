import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse, logging

import tensorflow as tf

from data import *
from models.mine import *
from utils.experiments import *


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='VisualVAE - Analysis')
    arg_parser.add_argument('vis_path', help='path to visual latents')
    arg_parser.add_argument('aud_path', help='path to auditive latents')
    arg_parser.add_argument('out_path', help='path to output')
    arg_parser.add_argument('--batch_size', type=int, default=200, help='batch size (default: 200)')
    arg_parser.add_argument('--epochs', type=int, default=100, help='number of training epochs (default: 100)')
    args = arg_parser.parse_args()

    # check if directory already exists
    if os.path.exists(args.out_path):
        print("[Warning] '%s' already exists." % (args.out_path,))
    # make necessary directories
    else:
        os.mkdir(args.out_path)
    
    setup_logging(os.path.join(args.out_path, 'results.log'))
    model_path = os.path.join(args.out_path, 'checkpoints')

    # set up MINE
    dataset = Latents(args.vis_path, args.aud_path)
    model = Mine(dataset.latent_dim, args.batch_size)
    model.build()

    # load data
    iterator = dataset.get_image_iterator(batch_size=args.batch_size)

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # training loop
        model.train(sess, iterator, None, epochs, model_path, args.out_path)
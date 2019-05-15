import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from data import *
from models.visual import *
from models.auditive import MusicVae
from models.synesthetic import SynestheticVae
from utils.experiments import *


if __name__ == '__main__':
    arg_parser = parse_arguments('SynestheticVAE - Training')
    arg_parser.add_argument('visvae_path', help='path to pre-trained VisualVAE (not required if training from scratch)')
    arg_parser.add_argument('musicvae_config', help='name of the MusicVAE model configuration (e.g. hierdec-mel_16bar)')
    arg_parser.add_argument('musicvae_path', help='path to MusicVAE model checkpoints')
    args = arg_parser.parse_args()
    model_path, tb_path, out_path, log_path = make_experiment_dir(args.exp_path)
    setup_logging(log_path)

    # set up auditive model
    music_vae = MusicVae(config_name=args.musicvae_config, batch_size=args.batch_size)
    # set up visual model
    if args.task == 'mnist':
        vis_model = MnistVae(latent_dim=music_vae.latent_dim, beta=args.beta, batch_size=args.batch_size)
        dataset = Mnist(split='train', data_path=args.data_path)
    elif args.task == 'cifar':
        vis_model = CifarVae(latent_dim=music_vae.latent_dim, beta=args.beta, batch_size=args.batch_size)
        dataset = Cifar(args.data_path)
    elif args.task == 'bam':
        vis_model = BamVae(latent_dim=music_vae.latent_dim, beta=args.beta, batch_size=args.batch_size)
        dataset = Bam(args.data_path)
    # set up synesthetic model
    model = SynestheticVae(visual_model=vis_model, auditive_model=music_vae, learning_rate=1e-3)
    model.build()

    # load data
    train_iterator, valid_iterator = dataset.get_train_image_iterators(batch_size=args.batch_size)

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # restore MusicVAE
        model.restore_auditive(tf_session=sess, path=args.musicvae_path)
        # restore visual model (if provided)
        if len(args.visvae_path) > 0:
            model.restore_visual(tf_session=sess, path=args.visvae_path)
        # training loop
        model.train(sess, train_iterator, valid_iterator, epochs, model_path, out_path)

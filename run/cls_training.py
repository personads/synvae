import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import tensorflow as tf

from data import *
from models.classifiers import *
from utils.experiments import *


if __name__ == '__main__':
    arg_parser = parse_arguments('VisualCNN - Training')
    args = arg_parser.parse_args()
    model_path, tb_path, out_path, log_path = make_experiment_dir(args.exp_path)
    setup_logging(log_path)

    # set up classifier
    if args.task == 'mnist':
        model = MnistCnn(batch_size=args.batch_size)
        dataset = Mnist(split='train', data_path=args.data_path)
    elif args.task == 'cifar':
        model = CifarCnn(batch_size=args.batch_size)
        dataset = Cifar(args.data_path)
    elif args.task == 'bam':
        model = BamCnn(batch_size=args.batch_size)
        dataset = Bam(args.data_path)
    model.build()

    # load data
    train_iterator, valid_iterator = dataset.get_train_iterators(batch_size=args.batch_size)

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # set up TensorBoard writer
        tf_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)
        # training loop
        model.train(sess, train_iterator, valid_iterator, epochs, model_path, out_path, tf_writer)
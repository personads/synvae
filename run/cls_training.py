import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import tensorflow as tf

from utils.experiments import *
from models.classifiers import MnistCnn, CifarCnn


if __name__ == '__main__':
    arg_parser = parse_arguments('VisualCNN - Training')
    arg_parser.add_argument('task', choices=['mnist', 'cifar'], help='name of the task (mnist, cifar)')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    args = arg_parser.parse_args()
    model_path, tb_path, out_path, log_path = make_experiment_dir(args.exp_path)
    setup_logging(log_path)

    # set up classifier
    if args.task == 'mnist':
        model = MnistCnn(batch_size=args.batch_size)
    elif args.task == 'cifar':
        model = CifarCnn(batch_size=args.batch_size)
    model.build()

    # load data
    images, labels, label_descs, num_labels = load_data(args.task, split='train', data_path=args.data_path)
    train_images, train_labels, valid_images, valid_labels = split_data(images, labels, args.task)

    # set up TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices({'images': train_images, 'labels': train_labels}).shuffle(train_images.shape[0]).batch(args.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    valid_dataset = tf.data.Dataset.from_tensor_slices({'images': valid_images, 'labels': valid_labels}).batch(args.batch_size)
    valid_iterator = valid_dataset.make_initializable_iterator()

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # set up TensorBoard writer
        tf_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)
        # training loop
        model.train(sess, train_iterator, valid_iterator, epochs, model_path, out_path, tf_writer)
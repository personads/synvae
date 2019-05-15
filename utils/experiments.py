import argparse, logging, os, pickle, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf

from data import *

def parse_arguments(exp_name):
    arg_parser = argparse.ArgumentParser(description=exp_name)
    arg_parser.add_argument('task', choices=['mnist', 'cifar','bam'], help='name of the task (mnist, cifar, bam)')
    arg_parser.add_argument('exp_path', help='path to experiment files (model checkpoints, TensorBoard logs, model outputs)')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('--beta', type=float, default=1., help='beta parameter for weighting KL-divergence (default: 1.0)')
    arg_parser.add_argument('--epochs', type=int, default=100, help='number of training epochs (default: 100)')
    arg_parser.add_argument('--init_epoch', type=int, default=0, help='epoch to resume at (default: 0)')
    arg_parser.add_argument('--batch_size', type=int, default=200, help='batch size for training and evaluation (default: 200)')
    return arg_parser


def make_experiment_dir(path):
    # check if directory already exists
    if os.path.exists(path):
        print("[Warning] '%s' already exists." % (path,))
    # make necessary directories
    else:
        os.mkdir(path)
    checkpoints_path = os.path.join(path, 'checkpoints')
    if not os.path.exists(checkpoints_path): os.mkdir(checkpoints_path)
    tensorboard_path = os.path.join(path, 'tensorboard')
    if not os.path.exists(tensorboard_path): os.mkdir(tensorboard_path)
    output_path = os.path.join(path, 'output')
    if not os.path.exists(output_path): os.mkdir(output_path)
    # make necessary paths
    log_path = os.path.join(path, 'experiment.log')
    return checkpoints_path, tensorboard_path, output_path, log_path


def setup_logging(log_path):
    log_format = '[%(levelname)s] %(asctime)s    %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    log_level = logging.INFO
    logging.basicConfig(filename=log_path, format=log_format, datefmt=log_datefmt, level=log_level)

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))


def load_data(data_name, split, data_path=None):
    # load data (initializes images and labels)
    if data_name == 'mnist':
        dataset = Mnist(split, data_path)
    elif data_name == 'cifar':
        dataset = Cifar(data_path)
    images = dataset.data
    labels = dataset.labels
    label_descs = dataset.label_descs
    num_labels = len(label_descs)

    logging.info("Loaded %d %s images from %s." % (images.shape[0], split, data_name.upper()))

    return images, labels, label_descs, num_labels

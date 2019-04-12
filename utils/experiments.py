import argparse, logging, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf

from data.cifar import Cifar

def parse_arguments(exp_name):
    arg_parser = argparse.ArgumentParser(description=exp_name)
    arg_parser.add_argument('task', choices=['mnist', 'cifar'], help='name of the task (mnist, cifar)')
    arg_parser.add_argument('exp_path', help='path to experiment files (model checkpoints, TensorBoard logs, model outputs)')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('--beta', type=float, default=1., help='beta parameter for weighting KL-divergence (default: 1.0)')
    arg_parser.add_argument('--epochs', type=int, default=100, help='number of training epochs (default: 100)')
    arg_parser.add_argument('--batch_size', type=int, default=200, help='batch size for training and evaluation (default: 200)')
    return arg_parser


def make_experiment_dir(path):
    # check if directory already exists
    if os.path.exists(path):
        print("[Error] '%s' already exists." % (path,))
        sys.exit()
    # make necessary directories
    os.mkdir(path)
    checkpoints_path = os.path.join(path, 'checkpoints')
    os.mkdir(checkpoints_path)
    tensorboard_path = os.path.join(path, 'tensorboard')
    os.mkdir(tensorboard_path)
    output_path = os.path.join(path, 'output')
    os.mkdir(output_path)
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
    images, labels = None, None
    label_descs, num_labels = None, None

    # load data (initializes images and labels)
    if data_name == 'mnist':
        # load MNIST
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        if split == 'train':
            images, labels = train_images, train_labels
        elif split == 'test':
            images, labels = test_images, test_labels
        images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
        # Normalizing the images to the range of [0., 1.]
        images /= 255.
        # Binarization
        images[images >= .5] = 1.
        images[images < .5] = 0.
        label_descs = [str(i) for i in range(10)]
        num_labels = len(label_descs)
    elif data_name == 'cifar':
        cifar = Cifar(data_path)
        images = cifar.data
        labels = cifar.labels
        label_descs = cifar.label_descs
        num_labels = len(label_descs)

    logging.info("Loaded %d %s images from %s." % (images.shape[0], split, data_name.upper()))

    return images, labels, label_descs, num_labels


def split_data(images, labels, task):
    split_idx = 0
    if task == 'mnist':
        split_idx = 50000
    elif task == 'cifar':
        split_idx = int(images.shape[0]*.8)
    train_images, train_labels = images[:split_idx], labels[:split_idx]
    valid_images, valid_labels = images[split_idx:], labels[split_idx:]
    logging.info("Split %s into %d training and %d validation images." % (task.upper(), train_images.shape[0], valid_images.shape[0]))
    return train_images, train_labels, valid_images, valid_labels
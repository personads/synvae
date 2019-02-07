import argparse, logging, os, sys

def parse_arguments(exp_name):
    arg_parser = argparse.ArgumentParser(description=exp_name)
    arg_parser.add_argument('exp_path', help='path to experiment files (model checkpoints, TensorBoard logs, model outputs)')
    arg_parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    arg_parser.add_argument('--batch_size', type=int, default=100, help='batch size for training and evaluation')
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
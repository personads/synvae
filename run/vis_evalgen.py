import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json, logging

import numpy as np
import tensorflow as tf

from collections import defaultdict, OrderedDict

from data import *
from utils.analysis import *
from utils.experiments import *
from models.visual import *


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='VisualVAE - Evaluation Task Generation')
    arg_parser.add_argument('task', choices=['mnist', 'cifar', 'bam'], help='name of the task')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('data_split', choices=['train', 'test'], default='test', help='data split (train, test (default))')
    arg_parser.add_argument('latent_path', help='path to latent vectors')
    arg_parser.add_argument('out_path', help='path to output')
    arg_parser.add_argument('--num_examples', type=int, default=4, help='number of examples for evaluation (default: 4)')
    arg_parser.add_argument('--num_tasks', type=int, default=20, help='number of tasks for evaluation (default: 20)')
    arg_parser.add_argument('--merge', action='store_true', help='merge classes with insufficient points into closest class')
    args = arg_parser.parse_args()
    
    # check if directory already exists
    if os.path.exists(args.out_path):
        print("[Error] '%s' already exists." % (args.out_path,))
        sys.exit()
    # make necessary directories
    os.mkdir(args.out_path)

    setup_logging(os.path.join(args.out_path, 'results.log'))

    # set up visual model
    if args.task == 'mnist':
        dataset = Mnist(split='test', data_path=args.data_path)
    elif args.task == 'cifar':
        dataset = Cifar(args.data_path)
    elif args.task == 'bam':
        dataset = Bam(args.data_path)
        dataset.filter_labels(['emotion_gloomy', 'emotion_happy', 'emotion_peaceful', 'emotion_scary'])
        dataset.filter_uncertain(round_up=False)
        dataset.make_multiclass()

    # load latent vectors
    latents = np.load(args.latent_path)
    dataset.labels = dataset.labels[:latents.shape[0]] # truncate if last batch was dropped

    # calculate means
    mean_latents = np.zeros([len(dataset.label_descs), latents.shape[1]])
    for c in range(len(dataset.label_descs)):
        lbl_idcs = np.where(dataset.labels == (c * np.ones_like(dataset.labels)))
        mean_latents[c] = np.mean(latents[lbl_idcs], axis=0)

    # reduce rare classes
    if args.merge:
        large_labels, small_labels = [], []
        for c in range(len(dataset.label_descs)):
            lbl_count = np.sum(dataset.labels == c)
            if lbl_count < (args.num_examples + args.num_tasks):
                small_labels.append(c)
            else:
                large_labels.append(c)
        logging.info("Found %d classes with insufficient datapoints." % len(small_labels))

        # reassign small labels to larger ones
        for c in small_labels:
            closest_labels, closest_dists = get_closest(mean_latents[c], mean_latents, [i for i in large_labels])
            lbl_idcs = np.where(dataset.labels == (c * np.ones_like(dataset.labels)))
            dataset.labels[lbl_idcs] = closest_labels[0]
            logging.info("Merged '%s' into '%s' with distance %.2f." % (dataset.label_descs[c], dataset.label_descs[closest_labels[0]], closest_dists[0]))
    
        for c in large_labels:
            lbl_idcs = np.where(dataset.labels == (c * np.ones_like(dataset.labels)))
            mean_latents[c] = np.mean(latents[lbl_idcs], axis=0)
    
        mean_latents = mean_latents[large_labels]

    # generate evaluation task
    logging.info("Exporting evaluation samples...")
    classes, examples, tasks = gen_eval_task(mean_latents, latents, dataset.labels, dataset.label_descs, args.num_examples, args.num_tasks)
    eval_config = OrderedDict([
        ('name', args.task.upper()),
        ('code', ''),
        ('data_path', ''),
        ('result_path', ''),
        ('classes', [dataset.label_descs[l] for l in classes]),
        ('examples', examples),
        ('tasks', tasks)
    ])
    eval_config_path = os.path.join(args.out_path, 'eval.json')
    with open(eval_config_path, 'w', encoding='utf8') as fop:
        json.dump(eval_config, fop)
    logging.info("Saved evaluation configuration with %d examples and %d tasks to '%s'." % (len(examples), len(tasks), eval_config_path))

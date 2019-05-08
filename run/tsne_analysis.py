import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse, json

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from matplotlib.offsetbox import *

from utils.experiments import load_data

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='tSNE Plot')
    arg_parser.add_argument('task', choices=['mnist', 'cifar'], help='name of the task (mnist, cifar)')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('data_split', choices=['train', 'test'], default='test', help='data split (train, test (default))')
    arg_parser.add_argument('latent_path', help='path to numpy latent vectors')
    arg_parser.add_argument('out_path', help='path to output')
    arg_parser.add_argument('--num_points', default=1000, help='number of data points to plot (default: 1000)')
    arg_parser.add_argument('--eval_task', default='', help='if path to eval JSON is provided, only data points from the task are plotted (default: None)')
    args = arg_parser.parse_args()

    # load data
    images, labels, label_descs, num_labels = load_data(args.task, split=args.data_split, data_path=args.data_path)
    cls_idx_map = [i for i in range(num_labels)]
    latents = np.load(args.latent_path)
    print("Loaded %d latents with dimensionality %d." % (latents.shape[0], latents.shape[1]))

    tSNE
    tsne_model = TSNE(n_components=2, verbose=True)
    tsne_latents = tsne_model.fit_transform(latents)

    # save transformation
    latents_name, latents_ext = os.path.splitext(os.path.basename(args.latent_path))
    tsne_path = os.path.join(args.out_path, '%s_tsne%s' % (latents_name, latents_ext))
    np.save(tsne_path, tsne_latents)
    print("Saved tSNE latents to '%s'." % tsne_path)
    # tsne_latents = np.load('/mnt/d/thesis/exp/tsne_plot/mnist_vae_01_latents_tsne.npy')

    # create subset
    if len(args.eval_task) > 0:
        eval_task = json.load(open(args.eval_task, 'r', encoding='utf8'))
        subset_idcs = eval_task['examples']
        for task in eval_task['tasks']:
            subset_idcs += task['options']
        subset_idcs = np.array(subset_idcs)
        cls_idx_map = [label_descs.index(d) for d in eval_task['classes']]
        label_descs = eval_task['classes']
        num_labels = len(label_descs)
        print("Loaded %d data points from eval task '%s'." % (len(subset_idcs), args.eval_task))
    else:
        subset_idcs = np.random.choice(tsne_latents.shape[0], args.num_points, replace=False)
        print("Reduced data points to random subset of size %d." % args.num_points)
    tsne_latents = tsne_latents[subset_idcs]
    labels = labels[subset_idcs]

    # init alphas
    alphas = np.zeros(tsne_latents.shape[0])

    # calculate means
    mean_latents = np.zeros([len(label_descs), tsne_latents.shape[1]])
    for c in range(num_labels):
        lbl_idcs = np.where(labels == (cls_idx_map[c] * np.ones_like(labels)))
        mean_latents[c] = np.mean(tsne_latents[lbl_idcs], axis=0)
        # calculate alphas
        dists = np.abs(tsne_latents[lbl_idcs] - mean_latents[c])
        dists = -np.sum(dists, axis=1)
        max_dist = np.max(dists)
        alphas[lbl_idcs] = np.clip(dists * (1 / max_dist), .3, None)

    fig, ax = plt.subplots()
    ax.scatter(tsne_latents[:, 0], tsne_latents[:, 1], alpha=0.)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot data points
    for i in range(tsne_latents.shape[0]):
        img = images[subset_idcs[i]].squeeze()
        ab = AnnotationBbox(OffsetImage(img, zoom=.5, cmap='gray', alpha=alphas[i]), (tsne_latents[i][0], tsne_latents[i][1]), frameon=False)
        ax.add_artist(ab)
    # plot means
    for c in range(len(label_descs)):
        bboxprops = dict(boxstyle='circle,pad=0.5', fc='black', ec='white')
        ax.text(mean_latents[c][0], mean_latents[c][1], label_descs[c], ha='center', color='white', fontweight='bold', size=8, alpha=.75, zorder=5, bbox=bboxprops)

    fig.savefig(os.path.join(args.out_path, 'tsne.pdf'))
    plt.show()
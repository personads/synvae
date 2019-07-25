import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse, json

import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from matplotlib.offsetbox import *
from PIL import Image

from utils.experiments import load_data

def load_image(path):
    img = Image.open(path)
    img = img.resize((32, 32))
    return np.array(img).squeeze()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='tSNE Plot')
    arg_parser.add_argument('task', choices=['mnist', 'cifar', 'bam'], help='name of the task (mnist, cifar)')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('data_split', choices=['train', 'test'], default='test', help='data split (train, test (default))')
    arg_parser.add_argument('latent_path', help='path to numpy latent vectors')
    arg_parser.add_argument('out_path', help='path to output')
    arg_parser.add_argument('--num_points', type=int, default=1000, help='number of data points to plot (default: 1000)')
    arg_parser.add_argument('--remove_outliers', type=float, default=0., help='removes outliers outside of n times the standard deviation (default: False)')
    arg_parser.add_argument('--eval_task', default='', help='if path to eval JSON is provided, only data points from the task are plotted (default: None)')
    arg_parser.add_argument('--tsne_latents', default='', help='if path to tSNE latents is provided, repeated projection will be skipped (default: None)')
    args = arg_parser.parse_args()

    # load data
    images, labels, label_descs, num_labels = load_data(args.task, split=args.data_split, data_path=args.data_path)
    cls_idx_map = [i for i in range(num_labels)]
    latents = np.load(args.latent_path)
    print("Loaded %d latents with dimensionality %d." % (latents.shape[0], latents.shape[1]))

    # tSNE
    if len(args.tsne_latents) > 0:
        tsne_latents = np.load(args.tsne_latents)
        print("Loaded tSNE latents from '%s'." % args.tsne_latents)
    else:
        tsne_model = TSNE(n_components=2, verbose=True)
        tsne_latents = tsne_model.fit_transform(latents)

        # save transformation
        latents_name, latents_ext = os.path.splitext(os.path.basename(args.latent_path))
        tsne_path = os.path.join(args.out_path, '%s_tsne%s' % (latents_name, latents_ext))
        np.save(tsne_path, tsne_latents)
        print("Saved tSNE latents to '%s'." % tsne_path)

    # create subset
    if len(args.eval_task) > 0:
        eval_task = json.load(open(args.eval_task, 'r', encoding='utf8'))
        eval_idcs = eval_task['examples'] + [i for task in eval_task['tasks'] for i in task['options']]
        other_idcs = [i for i in range(tsne_latents.shape[0]) if i not in eval_idcs]
        other_idcs = np.random.choice(other_idcs, args.num_points - len(eval_idcs), replace=False)
        subset_idcs = np.concatenate((other_idcs, eval_idcs))
        print("Loaded %d data points from eval task '%s'." % (len(eval_idcs), args.eval_task))
    else:
        subset_idcs = np.random.choice(tsne_latents.shape[0], args.num_points, replace=False)
        print("Reduced data points to random subset of size %d." % args.num_points)
    tsne_latents = tsne_latents[subset_idcs]
    labels = labels[subset_idcs]

    # shorten class names for BAM
    if args.task == 'bam':
        dmap = {'emotion_gloomy': 'g', 'emotion_happy': 'h', 'emotion_peaceful': 'p', 'emotion_scary': 's', 'unspecified': 'u'}
        label_descs = [''.join([dmap[e] for e in d.split('+')]) for d in label_descs]
        eval_task['classes'] = [''.join([dmap[e] for e in d.split('+')]) for d in eval_task['classes']]

    # init alphas
    alphas = np.zeros(tsne_latents.shape[0])

    # calculate means
    mean_latent = np.mean(tsne_latents, axis=0)
    std_latent = np.std(tsne_latents, axis=0)
    mean_latents = np.zeros([len(label_descs), tsne_latents.shape[1]])
    for c in range(num_labels):
        lbl_idcs = np.where(labels == (cls_idx_map[c] * np.ones_like(labels)))
        mean_latents[c] = np.mean(tsne_latents[lbl_idcs], axis=0)
        # calculate alphas
        if len(lbl_idcs[0]) > 1:
            dists = np.abs(tsne_latents[lbl_idcs] - mean_latents[c])
            dists = -np.sum(dists, axis=1)
            max_dist = np.max(dists)
            alphas[lbl_idcs] = np.clip(dists * (1 / max_dist), .3, None)
        else:
            alphas[lbl_idcs] = 0.

    # remove outliers
    if args.remove_outliers > 0:
        inlier_idcs = []
        for i in range(tsne_latents.shape[0]):
            if np.any(np.abs(tsne_latents[i] - mean_latent) > (std_latent * 3.)):
                continue
            inlier_idcs.append(i)
        tsne_latents = tsne_latents[inlier_idcs]
        labels = labels[inlier_idcs]
        alphas = alphas[inlier_idcs]
        subset_idcs = np.array(subset_idcs)[inlier_idcs]

    fig, ax = plt.subplots()
    ax.scatter(tsne_latents[:, 0], tsne_latents[:, 1], alpha=0., zorder=1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot data points
    for i in range(tsne_latents.shape[0]):
        # load image from path or directly
        if type(images[subset_idcs[i]]) is str:
            img = load_image(images[subset_idcs[i]])
        else:
            img = images[subset_idcs[i]].squeeze()
        if (len(args.eval_task) > 0) and (subset_idcs[i] in eval_idcs):
            bboxprops = dict(lw=2., ec='coral', alpha=.7)
            ab = AnnotationBbox(OffsetImage(img, zoom=.5, cmap='gray', alpha=.8), (tsne_latents[i][0], tsne_latents[i][1]), pad=0., bboxprops=bboxprops)
            # ax.text(tsne_latents[i][0], tsne_latents[i][1], str(subset_idcs[i]), ha='center', fontweight='bold', size=8, alpha=.75, zorder=5)
        else:
            ab = AnnotationBbox(OffsetImage(img, zoom=.5, cmap='gray', alpha=alphas[i]), (tsne_latents[i][0], tsne_latents[i][1]), pad=0., frameon=False)
        ax.add_artist(ab)
    # plot means
    for c in range(len(label_descs)):
        color = 'white'
        if (len(args.eval_task) > 0) and (label_descs[c] in eval_task['classes']):
            color = 'coral'
        bboxprops = dict(boxstyle='circle,pad=0.5', fc='black', ec=color)
        ax.text(mean_latents[c][0], mean_latents[c][1], label_descs[c], ha='center', color=color, fontweight='bold', size=8, alpha=.75, zorder=5, bbox=bboxprops)
    # plot mean connections (for eval)
    if len(args.eval_task) > 0:
        eval_mean_xs = [mean_latents[label_descs.index(eval_task['classes'][i%len(eval_task['classes'])])][0] for i in range(len(eval_task['classes'])+1)]
        eval_mean_ys = [mean_latents[label_descs.index(eval_task['classes'][i%len(eval_task['classes'])])][1] for i in range(len(eval_task['classes'])+1)]
        ax.plot(eval_mean_xs, eval_mean_ys, color='coral', alpha=.8, zorder=4)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_path, 'tsne.pdf'))
    plt.show()
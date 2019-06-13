import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging, random

import numpy as np
import tensorflow as tf
import sklearn.manifold.t_sne as tsne

from collections import defaultdict, OrderedDict
from scipy.spatial.distance import pdist, squareform


def calc_sims(latents):
    '''Calculate pairwise cosine similarities'''
    dot_mat = np.dot(latents, latents.T)
    nrm_mat = np.linalg.norm(latents, axis=1)
    mlt_mat = np.outer(nrm_mat, nrm_mat)
    sims = dot_mat / mlt_mat
    return sims

def calc_dists(latents):
    '''Calculate pairwise Euclidean distances'''
    return squareform(pdist(latents, 'euclidean'))


def get_closest(centroid, latents):
    dists = (latents - centroid)**2
    dists = np.sum(dists, axis=1)
    dists = np.sqrt(dists)

    dists = list(enumerate(dists))
    dists = sorted(dists, key=lambda el: el[1])
    latent_idcs, dists = zip(*dists)
    return np.array(latent_idcs), np.array(dists)


def calc_metrics(latents, labels, sims, num_labels, prec_ranks, sim_metric='euclidean'):
    # set up result arrays
    mean_latents = np.zeros([num_labels, latents.shape[1]])
    rel_sim_by_label = np.zeros(num_labels)
    oth_sim_by_label = np.zeros(num_labels)
    label_precision = defaultdict(lambda: np.zeros(num_labels))
    label_idcs = [[] for _ in range(num_labels)]

    # set up multi-class case
    if len(labels.shape) > 1:
        multi_labels = np.ones([num_labels, latents.shape[0]]) * -1
        for idx in range(latents.shape[0]):
            true_labels = np.where(labels[idx] == 1)[0]
            multi_labels[true_labels, idx] = true_labels
        label_counts = np.sum((multi_labels > -1), axis=-1)
        label_dist = (label_counts * 100) / np.sum(label_counts)
        logging.info("Set up multi-class evaluation with class distribution %s." % list(zip(label_counts.tolist(), label_dist.tolist())))

    for idx in range(latents.shape[0]):
        sys.stdout.write("\rCalculating metrics for %d/%d (%.2f%%)..." % (idx+1, latents.shape[0], ((idx+1)*100)/latents.shape[0]))
        sys.stdout.flush()

        true_labels = [labels[idx]] if len(labels.shape) < 2 else np.where(labels[idx] == 1.)[0] # get true label indices
        for lbl in true_labels:
            cur_labels = labels if len(labels.shape) < 2 else multi_labels[lbl] # set up labels for binary and multi-class

            # sort neighbours by similarity (without self)
            cur_sims = list(enumerate(np.concatenate((sims[idx, :idx], sims[idx, idx+1:]))))
            cur_sims = sorted(cur_sims, key=lambda el: el[1], reverse=(sim_metric == 'cosine'))
            # get sorted neighbours and similarities
            sim_idcs, sim_vals = zip(*cur_sims)
            sim_idcs, sim_vals = np.array(sim_idcs), np.array(sim_vals)

            # calculate average distances
            rel_idcs = np.where(cur_labels == (lbl * np.ones_like(cur_labels)))
            oth_idcs = np.where(cur_labels != (lbl * np.ones_like(cur_labels)))
            rel_avg_sim = np.mean(sims[idx][rel_idcs])
            oth_avg_sim = np.mean(sims[idx][oth_idcs])
            rel_sim_by_label[lbl] += rel_avg_sim
            oth_sim_by_label[lbl] += oth_avg_sim
            label_idcs[lbl].append(idx)

            # calculate precision/recall at top n
            for rank in prec_ranks:
                # get top n
                top_idcs, top_vals = sim_idcs[:rank], sim_vals[:rank]
                # count TP/FP and calculate precision
                tp = np.sum(cur_labels[top_idcs] == lbl)
                fp = np.sum(cur_labels[top_idcs] != lbl)
                precision = tp / (tp + fp)
                # store results
                label_precision[rank][lbl] += precision

    # compute mean latents
    for lbl in range(num_labels):
        mean_latents[lbl] = np.mean(latents[label_idcs[lbl]], axis=0)

    # average out metrics
    label_count = np.array([len(lbl_idcs) for lbl_idcs in label_idcs])
    rel_sim_by_label /= label_count
    oth_sim_by_label /= label_count
    for rank in prec_ranks:
        label_precision[rank] /= label_count

    logging.info("\rCalculated metrics for %d latents.%s" % (latents.shape[0], ' '*16))

    return mean_latents, rel_sim_by_label, oth_sim_by_label, label_precision


def calc_latent_kl(vis_latents, aud_latents, perplexity):
    logging.info("Calculating joint probability distribution of visual latent space...")
    vis_dists = calc_dists(vis_latents)
    vis_distr = tsne._joint_probabilities(distances=vis_dists, desired_perplexity=perplexity, verbose=True)
    logging.info("Calculating joint probability distribution of auditive latent space...")
    aud_dists = calc_dists(aud_latents)
    aud_distr = tsne._joint_probabilities(distances=aud_dists, desired_perplexity=perplexity, verbose=True)
    kl_va = 2.0 * np.dot(vis_distr, np.log(vis_distr / aud_distr))
    kl_av = 2.0 * np.dot(aud_distr, np.log(aud_distr / vis_distr))
    logging.info("Calculated KL divergences of audio-visual latent spaces with perplexity %d: %.2f VA / %.2f AV." % (perplexity, kl_va, kl_av))
    return kl_va, kl_av


def calc_cls_metrics(labels, predictions):
    # compute total accuracy
    pred_labels = np.argmax(predictions, axis=1)

    # compute accuracy, precision and recall by label
    label_accuracy = np.zeros(predictions.shape[1])
    label_precision = np.zeros(predictions.shape[1])
    label_recall = np.zeros(predictions.shape[1])
    for lbl in range(predictions.shape[1]):
        lbl_idcs = np.where(labels == (lbl * np.ones_like(labels)))
        oth_idcs = np.where(labels != (lbl * np.ones_like(labels)))
        tp = np.sum(pred_labels[lbl_idcs] == lbl)
        fp = np.sum(pred_labels[oth_idcs] == lbl)
        tn = np.sum(pred_labels[oth_idcs] != lbl)
        fn = np.sum(pred_labels[lbl_idcs] != lbl)
        label_precision[lbl] = tp / (tp + fp)
        label_recall[lbl] = tp / (tp + fn)
        label_accuracy[lbl] = (tp + tn) / (tp + fp + tn + fn)

    return np.mean(label_accuracy), label_precision, label_recall, label_accuracy


def calc_mltcls_metrics(labels, predictions):
    print("lbl:", labels.shape, labels[0], "pred:", predictions.shape, predictions[0])
    # round predictions to {0, 1}
    predictions = np.around(predictions)

    # compute accuracy, precision and recall by label
    label_accuracy = np.zeros(predictions.shape[1])
    label_precision = np.zeros(predictions.shape[1])
    label_recall = np.zeros(predictions.shape[1])
    for lbl in range(predictions.shape[1]):
        lbl_idcs = np.where(labels[:, lbl] == 1)
        oth_idcs = np.where(labels[:, lbl] == 0)
        tp = np.sum(predictions[lbl_idcs, lbl] == 1.)
        fp = np.sum(predictions[oth_idcs, lbl] == 1.)
        tn = np.sum(predictions[oth_idcs, lbl] == 0.)
        fn = np.sum(predictions[lbl_idcs, lbl] == 0.)
        label_precision[lbl] = tp / (tp + fp)
        label_recall[lbl] = tp / (tp + fn)
        label_accuracy[lbl] = (tp + tn) / (tp + fp + tn + fn)

    return np.mean(label_accuracy), label_precision, label_recall, label_accuracy


def log_metrics(label_descs, top_n, rel_sim_by_label, oth_sim_by_label, precision_by_label):
    logging.info("Overall metrics:")
    for label_idx, label in enumerate(label_descs):
        logging.info("  %s: %.2f P@%d, %.2f rel sim, %.2f oth sim" % (
            label, precision_by_label[label_idx], top_n,
            rel_sim_by_label[label_idx],
            oth_sim_by_label[label_idx]
            ))
    logging.info("Total (avg): %.2f P@%d, %.2f rel sim, %.2f oth sim" % (
            np.mean(precision_by_label), top_n,
            np.mean(rel_sim_by_label),
            np.mean(oth_sim_by_label)
        ))


def get_sorted_triplets(latents):
    # set up non-overlapping trios
    trio_keys = [
        tuple(sorted([i1, i2, i3]))
        for i1 in range(latents.shape[0])
            for i2 in range(latents.shape[0])
                for i3 in range(latents.shape[0])
                    if len(set([i1, i2, i3])) > 2
    ]
    # calculate trio similarities
    trio_sims = {}
    for trio_key in trio_keys:
        trio_sims[trio_key] = np.linalg.norm(latents[[trio_key[0]]] - latents[[trio_key[1]]])\
            + np.linalg.norm(latents[[trio_key[1]]] - latents[[trio_key[2]]])\
            + np.linalg.norm(latents[[trio_key[2]]] - latents[[trio_key[0]]])

    sorted_triplets = sorted(list(trio_sims.items()), key=lambda el: el[1], reverse=True)
    trio_keys, trio_dists = zip(*sorted_triplets)
    return trio_keys, trio_dists


def gen_eval_task(mean_latents, latents, labels, num_examples, num_tasks):
    # get triplet of means with largest distance between them
    trio_keys, trio_dists = get_sorted_triplets(mean_latents)
    eval_trio, eval_trio_dist = trio_keys[0], trio_dists[0]
    logging.info("Calculated mean triplet %s with cumulative Euclidean distance %.2f." % (str(eval_trio), eval_trio_dist))
    # get samples which lie closest to respective means
    trio_sample_idcs = np.zeros([3, num_examples + num_tasks], dtype=int)
    for tidx in range(3):
        # get indices of samples with same label as current mean
        rel_idcs = np.squeeze(np.where(labels == (eval_trio[tidx] * np.ones_like(labels))))
        rel_latents = latents[rel_idcs]
        # get closest latents of same label per mean
        closest_idcs, closest_dists = get_closest(mean_latents[eval_trio[tidx]], rel_latents)
        trio_sample_idcs[tidx] = rel_idcs[closest_idcs[:num_examples + num_tasks]]
        avg_dist = np.mean(closest_dists[:num_examples + num_tasks])
        logging.info("Calculated %d samples for mean %d with average distance %.2f." % (trio_sample_idcs[tidx].shape[0], eval_trio[tidx], avg_dist))
    # get examples
    example_idcs = sorted(np.random.choice((num_examples + num_tasks), num_examples, replace=False))
    examples = np.squeeze(trio_sample_idcs[:,example_idcs].flatten())
    examples = examples.tolist()
    trio_sample_idcs[:,example_idcs] = -1
    # get tasks
    task_idcs = np.where(trio_sample_idcs >= 0.)
    task_trios = np.reshape(trio_sample_idcs[task_idcs], [3, num_tasks])
    task_trios = [task_trios[:, i].tolist() for i in range(num_tasks)]
    # randomly select truths for tasks
    tasks = []
    for trio in task_trios:
        truth_idx = random.randint(0, 2)
        tasks.append(OrderedDict([
            ('truth', truth_idx),
            ('options', trio)
        ]))
    return eval_trio, examples, tasks

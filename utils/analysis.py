import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging, math, multiprocessing, random

import numpy as np
import tensorflow as tf
import sklearn
import sklearn.manifold.t_sne as tsne

from collections import defaultdict, OrderedDict


def calc_sims(latents):
    '''Calculate pairwise cosine similarities'''
    dot_mat = np.dot(latents, latents.T)
    nrm_mat = np.linalg.norm(latents, axis=1)
    mlt_mat = np.outer(nrm_mat, nrm_mat)
    sims = dot_mat / mlt_mat
    return sims

def calc_dists(latents, split=10000):
    '''Calculate pairwise Euclidean distances'''
    # split high memory load
    if latents.shape[0] > split:
        dists = np.zeros([latents.shape[0], latents.shape[0]])
        for i in range(math.ceil(latents.shape[0]/split)):
            start_idx, end_idx = (i * split), ((i + 1) * split)
            subset = latents[start_idx:end_idx]
            dists[start_idx:end_idx] = sklearn.metrics.pairwise_distances(subset, latents, metric='euclidean', n_jobs=multiprocessing.cpu_count())
    # otherwise, calculate in one go
    else:
        dists = sklearn.metrics.pairwise_distances(latents, latents, metric='euclidean', n_jobs=multiprocessing.cpu_count())
    return dists


def get_closest(centroid, latents, rel_idcs):
    rel_latents = latents[rel_idcs]
    dists = (rel_latents - centroid)**2
    dists = np.sum(dists, axis=1)
    dists = np.sqrt(dists)

    dists = [(rel_idcs[i], d) for i, d in enumerate(dists)] # re-map to global latent indices
    dists = sorted(dists, key=lambda el: el[1])
    latent_idcs, dists = zip(*dists)
    return np.array(latent_idcs, dtype=int), np.array(dists)


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
        label_dist = np.around((label_counts * 100) / np.sum(label_counts), decimals=2)
        logging.info("Set up multi-class evaluation with class distribution %s." % list(zip(label_counts.tolist(), label_dist.tolist())))

    for idx in range(latents.shape[0]):
        sys.stdout.write("\rCalculating metrics for %d/%d (%.2f%%)..." % (idx+1, latents.shape[0], ((idx+1)*100)/latents.shape[0]))
        sys.stdout.flush()

        true_labels = [labels[idx]] if len(labels.shape) < 2 else np.where(labels[idx] == 1.)[0] # get true label indices
        for lbl in true_labels:
            cur_labels = labels if len(labels.shape) < 2 else multi_labels[lbl] # set up labels for binary and multi-class

            # sort neighbours by similarity (without self)
            cur_sims = [(i, s) for i, s in enumerate(sims[idx]) if i != idx]
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
    label_counts = np.array([len(lbl_idcs) for lbl_idcs in label_idcs])
    rel_sim_by_label /= label_counts
    oth_sim_by_label /= label_counts
    for rank in prec_ranks:
        label_precision[rank] /= label_counts

    logging.info("\rCalculated metrics for %d latents.%s" % (latents.shape[0], ' '*16))

    return mean_latents, rel_sim_by_label, oth_sim_by_label, label_precision, label_counts


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


def log_metrics(label_descs, top_n, rel_sim_by_label, oth_sim_by_label, precision_by_label, label_counts):
    logging.info("Overall metrics:")
    for label_idx, label in enumerate(label_descs):
        logging.info("  %s: %.2f P@%d, %.2f rel sim, %.2f oth sim" % (
            label, precision_by_label[label_idx], top_n,
            rel_sim_by_label[label_idx],
            oth_sim_by_label[label_idx]
            ))
    logging.info("Total (avg): %.2f P@%d, %.2f rel sim, %.2f oth sim" % (
            np.sum(precision_by_label * (label_counts/np.sum(label_counts))), top_n,
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


def get_unique_samples(closest_idcs, closest_dists):
    # sort distances globally
    idx_dist_map = [
        ([row, col], closest_dists[row, col])
        for row in range(closest_dists.shape[0])
        for col in range(closest_dists.shape[1])
        if closest_dists[row, col] >= 0]
    idx_dist_map = sorted(idx_dist_map, key=lambda el: el[1])

    # go through samples in globally sorted order
    sample_idcs = np.ones_like(closest_idcs, dtype=int) * -1
    for seek, dist in idx_dist_map:
        cur_idx = closest_idcs[seek[0], seek[1]]
        # check if idx is already used for mean which it is closer to
        if cur_idx in sample_idcs:
            continue
        # insert at leftmost position for appropriate class
        sample_idcs[seek[0], np.where(sample_idcs[seek[0]] == -1)[0][0]] = cur_idx

    return sample_idcs


def gen_eval_task(mean_latents, latents, labels, label_descs, num_examples, num_tasks):
    # get triplet of means with largest distance between them
    trio_keys, trio_dists = get_sorted_triplets(mean_latents)
    for t, d in zip(trio_keys, trio_dists):
        logging.info("Calculated mean triplet (%s) with cumulative Euclidean distance %.2f." % (', '.join([label_descs[l] for l in t]), d))
    # iterate over triplets (in case one set has insufficient amounts of data)
    for eval_trio, eval_trio_dist in zip(trio_keys, trio_dists):
        logging.info("Calculated mean triplet (%s) with cumulative Euclidean distance %.2f." % (', '.join([label_descs[l] for l in eval_trio]), eval_trio_dist))
        # get samples which lie closest to respective means
        closest_idcs = np.ones([3, latents.shape[0]], dtype=int) * -1
        closest_dists = np.ones([3, latents.shape[0]]) * -1
        is_valid_trio = True
        for tidx in range(3):
            # get indices of samples with same label as current mean
            if len(labels.shape) > 1:
                rel_idcs = np.squeeze(np.where(labels[:, eval_trio[tidx]] == 1.))
            else:
                rel_idcs = np.squeeze(np.where(labels == (eval_trio[tidx] * np.ones_like(labels))))
            # check if class has enough samples to generate tasks
            if (len(rel_idcs.shape) < 1) or (rel_idcs.shape[0] < (num_examples + num_tasks)):
                is_valid_trio = False
                break
            # get closest latents of same label per mean
            cur_closest_idcs, cur_closest_dists = get_closest(mean_latents[eval_trio[tidx]], latents, rel_idcs)
            closest_idcs[tidx, :cur_closest_idcs.shape[0]] = cur_closest_idcs
            closest_dists[tidx, :cur_closest_dists.shape[0]] = cur_closest_dists
        # exit loop if sufficient amounts are available
        if is_valid_trio:
            trio_sample_idcs = get_unique_samples(closest_idcs, closest_dists)
            break
        # skip to next trio if current one is insufficient
        else:
            logging.error("[Error] Not enough data to generate task based on classes %s." % (eval_trio,))
            continue

    # truncate trio sample idcs
    trio_sample_idcs = trio_sample_idcs[:, :(num_examples + num_tasks + 1)]

    # get examples (randomly choose from available data)
    example_idcs = sorted(np.random.choice((num_examples + num_tasks + 1), num_examples, replace=False))
    examples = np.squeeze(trio_sample_idcs[:,example_idcs].flatten())
    examples = examples.tolist()
    trio_sample_idcs[:,example_idcs] = -1

    # get tasks
    task_idcs = np.where(trio_sample_idcs >= 0.)
    task_trios = np.reshape(trio_sample_idcs[task_idcs], [3, -1])
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

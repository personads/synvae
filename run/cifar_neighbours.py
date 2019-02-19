import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from data.cifar import Cifar
from models.visual import CifarVae

from experiments import *


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='CIFAR10 Nearest Neighbours')
    arg_parser.add_argument('model_path', help='path to model')
    arg_parser.add_argument('cifar_path', help='path to CIFAR10')
    arg_parser.add_argument('out_path', help='path to output')
    args = arg_parser.parse_args()

    setup_logging(os.path.join(args.out_path, 'results.log'))

    batch_size = 100
    export_step = 200
    top_n = 10
    # set up model
    cifar_vae = CifarVae(latent_dim=512, batch_size=batch_size)
    # load CIFAR
    cifar = Cifar(args.cifar_path)
    logging.info("Loaded %d images from CIFAR10." % (cifar.data.shape[0]))
    # set up TF datasets
    num_batches = cifar.data.shape[0] // batch_size + 1
    tf_dataset = tf.data.Dataset.from_tensor_slices(cifar.data).batch(batch_size)
    tf_iterator = tf_dataset.make_initializable_iterator()
    logging.info("Loaded data into TensorFlow.")

    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # restore session
        cifar_vae.restore(sess, args.model_path)

        # encode in batches
        next_op = tf_iterator.get_next()
        sess.run(tf_iterator.initializer)
        reconstructions, latents = None, None
        # iterate over batches
        avg_loss = 0.
        batch_idx = 0
        while True:
            try:
                sys.stdout.write("\rencoding batch %d..." % (batch_idx))
                sys.stdout.flush()
                batch = sess.run(next_op)
                batch_idx += 1
                cur_loss, cur_recons, cur_latents = sess.run([cifar_vae.loss, cifar_vae.reconstructions, cifar_vae.latents], feed_dict={cifar_vae.images: batch})
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                # append to result
                if (reconstructions is None) or (latents is None):
                    reconstructions, latents = cur_recons, cur_latents
                else:
                    reconstructions = np.concatenate((reconstructions, cur_recons), axis=0)
                    latents = np.concatenate((latents, cur_latents), axis=0)
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
        logging.info("\rencoded %d batches with avg_loss %.2f, %d reconstructions and %d latent vectors." % (batch_idx, avg_loss, reconstructions.shape[0], latents.shape[0]))

        # calculate similarities
        logging.info("Calculating similarities...")
        dot_mat = np.dot(latents, latents.T)
        nrm_mat = np.linalg.norm(latents, axis=1)
        mlt_mat = np.outer(nrm_mat, nrm_mat)
        sims = dot_mat / mlt_mat

        rel_sim_by_label = np.zeros(len(cifar.label_descs))
        oth_sim_by_label = np.zeros(len(cifar.label_descs))
        precision_by_label = np.zeros(len(cifar.label_descs))
        recall_by_label = np.zeros(len(cifar.label_descs))
        label_count = np.zeros(len(cifar.label_descs))

        logging.info("Calculating metrics (exporting for %d images)..." % (cifar.data.shape[0]//export_step))
        for img_idx in range(cifar.data.shape[0]):
            img_cls = cifar.labels[img_idx]
            # sort neighbours by similarity
            cur_sims = list(enumerate(sims[img_idx]))
            cur_sims = sorted(cur_sims, key=lambda el: el[1], reverse=True)
            # get n most similar
            sim_img_idcs = cur_sims[:top_n+1]
            top_idcs, top_sims = zip(*sim_img_idcs)
            top_idcs, top_sims = np.array(top_idcs), np.array(top_sims)

            # calculate average distances
            rel_idcs = np.where(cifar.labels == (img_cls * np.ones_like(cifar.labels)))
            oth_idcs = np.where(cifar.labels != (img_cls * np.ones_like(cifar.labels)))
            rel_avg_sim = np.mean(sims[img_idx][rel_idcs])
            oth_avg_sim = np.mean(sims[img_idx][oth_idcs])
            rel_sim_by_label[img_cls] += rel_avg_sim
            oth_sim_by_label[img_cls] += oth_avg_sim
            label_count[img_cls] += 1

            # calculate precision/recall at top n
            tp = np.sum(cifar.labels[top_idcs] == img_cls)
            fp = np.sum(cifar.labels[top_idcs] != img_cls)
            precision = tp / (tp + fp)
            recall = tp / len(rel_idcs)
            precision_by_label[img_cls] += precision
            recall_by_label[img_cls] += recall

            # export nearest neighbours every n steps
            if img_idx % export_step == 0:
                # save original
                cifar_vae.save_image(cifar.data[img_idx], os.path.join(args.out_path, str(img_idx) + '_orig.png'))
                # export similar images
                sim_img_top = 0
                for sim_img_idx, cur_sim in sim_img_idcs:
                    cifar_vae.save_image(cifar.data[sim_img_idx], os.path.join(args.out_path, '%d_%d_%d_%.2f.png' % (img_idx, sim_img_top, sim_img_idx, cur_sim)))
                    sim_img_top += 1

        # average out metrics
        rel_sim_by_label /= label_count
        oth_sim_by_label /= label_count
        precision_by_label /= label_count
        recall_by_label /= label_count

        logging.info("Overall metrics:")
        for label_idx, label in enumerate(cifar.label_descs):
            logging.info("  %s: %.2f P@%d, %.2f R@%d, %.2f rel sim, %.2f oth sim" % (
                label, precision_by_label[label_idx], top_n,
                recall_by_label[label_idx], top_n,
                rel_sim_by_label[label_idx], oth_sim_by_label[label_idx]
                ))
        logging.info("Total (avg): %.2f P@%d, %.2f R@%d, %.2f rel sim, %.2f oth sim" % (
                np.mean(precision_by_label[label_idx]), top_n,
                np.mean(recall_by_label[label_idx]), top_n,
                np.mean(rel_sim_by_label[label_idx]), np.mean(oth_sim_by_label[label_idx])
            ))



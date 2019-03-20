import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from models.visual import MnistVae, CifarVae
from models.auditive import MusicVae
from models.synesthetic import SynestheticVae

from experiments import *

def calc_sims(latents):
    dot_mat = np.dot(latents, latents.T)
    nrm_mat = np.linalg.norm(latents, axis=1)
    mlt_mat = np.outer(nrm_mat, nrm_mat)
    sims = dot_mat / mlt_mat
    return sims

def calc_metrics(data, labels, sims, num_labels, export_step, export_fn, export_ext):
    rel_sim_by_label = np.zeros(len(num_labels))
    oth_sim_by_label = np.zeros(len(num_labels))
    precision_by_label = np.zeros(len(num_labels))
    label_count = np.zeros(len(num_labels))

    for idx in range(data.shape[0]):
        lbl = labels[idx]
        # sort neighbours by similarity
        cur_sims = list(enumerate(sims[idx]))
        cur_sims = sorted(cur_sims, key=lambda el: el[1], reverse=True)
        # get n most similar
        sim_img_idcs = cur_sims[:top_n+1]
        top_idcs, top_sims = zip(*sim_img_idcs)
        top_idcs, top_sims = np.array(top_idcs), np.array(top_sims)

        # calculate average distances
        rel_idcs = np.where(labels == (lbl * np.ones_like(labels)))
        oth_idcs = np.where(labels != (lbl * np.ones_like(labels)))
        rel_avg_sim = np.mean(sims[idx][rel_idcs])
        oth_avg_sim = np.mean(sims[idx][oth_idcs])
        rel_sim_by_label[lbl] += rel_avg_sim
        oth_sim_by_label[lbl] += oth_avg_sim
        label_count[lbl] += 1

        # calculate precision/recall at top n
        tp = np.sum(labels[top_idcs] == lbl)
        fp = np.sum(labels[top_idcs] != lbl)
        precision = tp / (tp + fp)
        precision_by_label[lbl] += precision

    # average out metrics
    rel_sim_by_label /= label_count
    oth_sim_by_label /= label_count
    precision_by_label /= label_count

    return rel_sim_by_label, oth_sim_by_label, precision_by_label

def log_metrics(label_descs, rel_sim_by_label, oth_sim_by_label, precision_by_label):
    logging.info("Overall metrics:")
    for label_idx, label in enumerate(label_descs):
        logging.info("  %s: %.2f P@%d, %.2f rel sim, %.2f oth sim" % (
            label, precision_by_label[label_idx], top_n,
            rel_sim_by_label[label_idx],
            oth_sim_by_label[label_idx]
            ))
    logging.info("Total (avg): %.2f P@%d, %.2f rel sim, %.2f oth sim" % (
            np.mean(precision_by_label[label_idx]), top_n,
            np.mean(rel_sim_by_label[label_idx]),
            np.mean(oth_sim_by_label[label_idx])
        ))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='SynVAE - Analysis')
    arg_parser.add_argument('task', choices=['mnist', 'cifar'], help='name of the task (mnist, cifar)')
    arg_parser.add_argument('musicvae_config', choices=['cat-mel_2bar_big', 'hierdec-mel_16bar'], help='name of the MusicVAE model configuration (e.g. hierdec-mel_16bar)')
    arg_parser.add_argument('model_path', help='path to SynVAE model')
    arg_parser.add_argument('data_path', help='path to data (required for CIFAR)')
    arg_parser.add_argument('out_path', help='path to output')
    arg_parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    args = arg_parser.parse_args()

    # check if directory already exists
    if os.path.exists(args.out_path):
        print("[Error] '%s' already exists." % (args.out_path,))
        sys.exit()
    # make necessary directories
    os.mkdir(args.out_path)
    
    setup_logging(os.path.join(args.out_path, 'results.log'))

    # set up auditive model
    music_vae = MusicVae(config_name=args.musicvae_config, batch_size=args.batch_size)
    # set up visual model
    if args.task == 'mnist':
        visual_vae = MnistVae(latent_dim=music_vae.latent_dim, batch_size=args.batch_size)
    elif args.task == 'cifar':
        visual_vae = CifarVae(latent_dim=music_vae.latent_dim, batch_size=args.batch_size)
    # set up synesthetic model
    model = SynestheticVae(visual_model=visual_vae, auditive_model=music_vae, learning_rate=1e-4)
    model.build()

    # load data (initializes images and labels)
    if args.task == 'mnist':
        # load MNIST
        (_, _), (images, labels) = tf.keras.datasets.mnist.load_data()
        images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
        # Normalizing the images to the range of [0., 1.]
        images /= 255.
        # Binarization
        images[images >= .5] = 1.
        images[images < .5] = 0.
        label_descs = [str(i) for i in range(10)]
        num_labels = len(label_descs)
        logging.info("Loaded %d test images from MNIST." % (images.shape[0],))
    elif args.task == 'cifar':
        if args.data_path is None:
            arg_parser.error("CIFAR task requires 'data_path' argument.")
        cifar = Cifar(args.data_path)
        images = cifar.data
        labels = cifar.labels
        label_descs = cifar.label_descs
        num_labels = len(label_descs)
        logging.info("Loaded %d images from CIFAR10." % (images.shape[0]))
    
    # set up TF datasets
    dataset = tf.data.Dataset.from_tensor_slices(images).batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_op = iterator.get_next()

    # inference
    with tf.Session() as sess:
        # initialize variables and dataset iterator
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        # restore MusicVAE
        model.restore(tf_session=sess, path=args.model_path)

        # encode in batches
        audios, reconstructions, vis_latents, aud_latents = None, None, None, None
        avg_loss = 0.
        batch_idx = 0
        # iterate over batches
        while True:
            try:
                # get next batch
                sys.stdout.write("\rEncoding batch %d..." % (batch_idx))
                sys.stdout.flush()
                batch = sess.run(next_op)
                batch_idx += 1

                # inference step
                epsilons = np.zeros((batch.shape[0], model.latent_dim))
                temperature = 0.5
                cur_loss, cur_audios, cur_recons, cur_vis_latents, cur_aud_latents = sess.run([
                        model.loss, model.audios, model.reconstructions, model.vis_latents, model.aud_latents
                    ], feed_dict={
                        model.images: batch, model.epsilons: epsilons, model.temperature: temperature
                    })

                # append to result
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                if (reconstructions is None) or (latents is None):
                    audios, reconstructions, vis_latents, aud_latents = cur_audios, cur_recons, cur_vis_latents, cur_aud_latents
                else:
                    audios = np.concatenate((audios, cur_audios), axis=0)
                    reconstructions = np.concatenate((reconstructions, cur_recons), axis=0)
                    vis_latents = np.concatenate((vis_latents, cur_vis_latents), axis=0)
                    aud_latents = np.concatenate((aud_latents, cur_aud_latents), axis=0)
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
    logging.info("\rEncoded %d batches with avg_loss %.2f, %d audios, %d reconstructions and %d latent vectors."
        % (batch_idx, avg_loss, audios.shape[0], reconstructions.shape[0], latents.shape[0]))

    logging.info("Saving outputs...")
    for idx in range(images.shape[0]):
        model.vis_model.save_image(images[idx], os.path.join(args.out_path, str(idx) + '_orig.png'))
        model.vis_model.save_image(reconstructions[idx], os.path.join(args.out_path, str(idx) + '_recon.png'))
        model.aud_model.save_midi(audios[idx], os.path.join(args.out_path, str(idx) + '_audio.midi'))
        sys.stdout.write("\rSaved %d/%d (%.2f%%)..." % (idx+1, images.shape[0], ((idx+1)*100)/images.shape[0]))
        sys.stdout.flush()
    logging.info("\rSaved %d images, audios and reconstructions." % images.shape[0])

    logging.info("Calculating similarities...")
    vis_sims = calc_sims(vis_latents)
    aud_sims = calc_sims(aud_latents)
    np.save(os.path.join(args.out_path, str(idx) + 'vis_sims.npy'), vis_sims)
    np.save(os.path.join(args.out_path, str(idx) + 'aud_sims.npy'), aud_sims)
    logging.info("Saved audio and visual similarities to %s." % os.path.join(args.out_path, str(idx) + '*_sims.npy'))

    logging.info("Calculating metrics for visual latents...")
    rel_sim_by_label, oth_sim_by_label, precision_by_label = calc_metrics(images, labels, vis_sims, num_labels)
    log_metrics(label_descs, rel_sim_by_label, oth_sim_by_label, precision_by_label)
    logging.info("Calculating metrics for auditive latents...")
    rel_sim_by_label, oth_sim_by_label, precision_by_label = calc_metrics(images, labels, aud_sims, num_labels)
    log_metrics(label_descs, rel_sim_by_label, oth_sim_by_label, precision_by_label)
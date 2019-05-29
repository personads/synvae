import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from collections import defaultdict

from data import *
from utils.analysis import *
from utils.experiments import *
from models.visual import MnistVae, CifarVae
from models.auditive import MusicVae
from models.synesthetic import SynestheticVae


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='SynVAE - Analysis')
    arg_parser.add_argument('task', choices=['mnist', 'cifar', 'bam'], help='name of the task (mnist, cifar, bam)')
    arg_parser.add_argument('musicvae_config', choices=['cat-mel_2bar_big', 'hierdec-mel_4bar', 'hierdec-mel_8bar' 'hierdec-mel_16bar'], help='name of the MusicVAE model configuration (e.g. hierdec-mel_16bar)')
    arg_parser.add_argument('model_path', help='path to SynVAE model')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('data_split', choices=['train', 'test'], default='test', help='data split (train, test (default))')
    arg_parser.add_argument('out_path', help='path to output')
    arg_parser.add_argument('--beta', type=float, default=1., help='beta parameter for weighting KL-divergence (default: 1.0)')
    arg_parser.add_argument('--batch_size', type=int, default=200, help='batch size (default: 200)')
    arg_parser.add_argument('--ranks', default='1,5,10', help='precision ranks to use during evaluation (default: "1,5,10")')
    arg_parser.add_argument('--kl', action='store_true', help='compute approximate audio-visual KL-divergence')
    arg_parser.add_argument('--perplexity', type=int, default=30, help='perplexity of distributions used to approximate the data space (default: 30)')
    arg_parser.add_argument('--export', action='store_true', help='export original samples and reconstructions')
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
        visual_vae = MnistVae(latent_dim=music_vae.latent_dim, beta=args.beta, batch_size=args.batch_size)
        dataset = Mnist(split='test', data_path=args.data_path)
    elif args.task == 'cifar':
        visual_vae = CifarVae(latent_dim=music_vae.latent_dim, beta=args.beta, batch_size=args.batch_size)
        dataset = Cifar(args.data_path)
    elif args.task == 'bam':
        visual_vae = BamVae(latent_dim=music_vae.latent_dim, beta=args.beta, batch_size=args.batch_size)
        dataset = Bam(args.data_path)
    # set up synesthetic model
    model = SynestheticVae(visual_model=visual_vae, auditive_model=music_vae, learning_rate=1e-4)
    model.build()

    # load data
    iterator = dataset.get_image_iterator(batch_size=args.batch_size)
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
        avg_mse = 0.
        avg_kl = 0.
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
                temperature = 0.5
                cur_loss, cur_mse, cur_kl, cur_audios, cur_recons, cur_vis_latents, cur_aud_latents = sess.run([
                        model.loss, model.vis_model.recon_loss, model.vis_model.latent_loss, model.audios, model.reconstructions, model.vis_latents, model.aud_latents
                    ], feed_dict={
                        model.images: batch, model.temperature: temperature
                    })

                # append to result
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                avg_mse = ((avg_mse * (batch_idx - 1)) + cur_mse) / batch_idx
                avg_kl = ((avg_kl * (batch_idx - 1)) + cur_kl) / batch_idx
                if batch_idx == 1:
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
    logging.info("\rEncoded %d batches with average losses (All: %.2f | MSE: %.2f | KL: %.2f), %d audios, %d reconstructions, %d visual latent vectors and %d auditive latent vectors."
        % (batch_idx, avg_loss, avg_mse, avg_kl, audios.shape[0], reconstructions.shape[0], vis_latents.shape[0], aud_latents.shape[0]))

    if args.export:
        logging.info("Saving outputs...")
        for idx in range(images.shape[0]):
            model.vis_model.save_image(images[idx].squeeze(), os.path.join(args.out_path, str(idx) + '_orig.png'))
            model.vis_model.save_image(reconstructions[idx].squeeze(), os.path.join(args.out_path, str(idx) + '_recon.png'))
            model.aud_model.save_midi(audios[idx], os.path.join(args.out_path, str(idx) + '_audio.mid'))
            sys.stdout.write("\rSaved %d/%d (%.2f%%)..." % (idx+1, images.shape[0], ((idx+1)*100)/images.shape[0]))
            sys.stdout.flush()
        logging.info("\rSaved %d images, audios and reconstructions." % images.shape[0])

    if args.kl:
        logging.info("Calculating KL divergence between latents (perplexity: %d)..." % args.perplexity)
        kl_va, kl_av = calc_latent_kl(vis_latents, aud_latents, perplexity=args.perplexity)

    logging.info("Calculating similarities...")
    vis_sims = calc_dists(vis_latents)
    aud_sims = calc_dists(aud_latents)

    # parse precision ranks
    prec_ranks = [int(r) for r in args.ranks.split(',')]

    logging.info("Calculating metrics for visual latents...")
    vis_mean_latents, rel_sim_by_label, oth_sim_by_label, label_precision = calc_metrics(vis_latents, labels, vis_sims, len(dataset.label_descs), prec_ranks, sim_metric='euclidean')
    for rank in prec_ranks:
        log_metrics(dataset.label_descs, rank, rel_sim_by_label, oth_sim_by_label, label_precision[rank])

    logging.info("Calculating metrics for auditive latents...")
    aud_mean_latents, rel_sim_by_label, oth_sim_by_label, label_precision = calc_metrics(aud_latents, labels, aud_sims, len(dataset.label_descs), prec_ranks, sim_metric='euclidean')
    for rank in prec_ranks:
        log_metrics(dataset.label_descs, rank, rel_sim_by_label, oth_sim_by_label, label_precision[rank])

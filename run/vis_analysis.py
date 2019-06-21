import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import numpy as np
import tensorflow as tf

from collections import defaultdict, OrderedDict

from data import *
from utils.analysis import *
from utils.experiments import *
from models.visual import *


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='VisualVAE - Analysis')
    arg_parser.add_argument('task', choices=['mnist', 'cifar', 'bam'], help='name of the task')
    arg_parser.add_argument('model_path', help='path to VisualVAE model')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('data_split', choices=['train', 'test'], default='test', help='data split (train, test (default))')
    arg_parser.add_argument('out_path', help='path to output')
    arg_parser.add_argument('--beta', type=float, default=1., help='beta parameter for weighting KL-divergence (default: 1.0)')
    arg_parser.add_argument('--batch_size', type=int, default=200, help='batch size (default: 200)')
    arg_parser.add_argument('--ranks', default='1,5,10', help='precision ranks to use during evaluation (default: "1,5,10")')
    arg_parser.add_argument('--export_data', action='store_true', help='export original samples and reconstructions')
    arg_parser.add_argument('--export_latents', action='store_true', help='export latent vectors')
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
        model = MnistVae(latent_dim=50, beta=args.beta, batch_size=args.batch_size)
        dataset = Mnist(split='test', data_path=args.data_path)
    elif args.task == 'cifar':
        model = CifarVae(latent_dim=512, beta=args.beta, batch_size=args.batch_size)
        dataset = Cifar(args.data_path)
    elif args.task == 'bam':
        model = BamVae(latent_dim=512, beta=args.beta, batch_size=args.batch_size)
        dataset = Bam(args.data_path)
        dataset.filter_labels(['emotion_gloomy', 'emotion_happy', 'emotion_peaceful', 'emotion_scary'])
        dataset.filter_uncertain()
        dataset.make_multiclass()
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
        
        latents = None
        # iterate over batches
        avg_loss = 0.
        avg_mse = 0.
        avg_kl = 0.
        batch_idx = 0
        while True:
            try:
                sys.stdout.write("\rEncoding batch %d..." % (batch_idx))
                sys.stdout.flush()
                batch = sess.run(next_op)
                batch_idx += 1
                cur_loss, cur_mse, cur_kl, cur_recons, cur_latents = sess.run([model.loss, model.recon_loss, model.latent_loss, model.reconstructions, model.latents], feed_dict={model.images: batch})
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                avg_mse = ((avg_mse * (batch_idx - 1)) + cur_mse) / batch_idx
                avg_kl = ((avg_kl * (batch_idx - 1)) + cur_kl) / batch_idx
                # append to result
                if batch_idx == 1:
                    latents = cur_latents
                else:
                    latents = np.concatenate((latents, cur_latents), axis=0)

                if args.export_data:
                    for idx in range(batch.shape[0]):
                        data_idx = ((batch_idx - 1) * args.batch_size) + idx
                        model.save_image(batch[idx].squeeze(), os.path.join(args.out_path, str(data_idx) + '_orig.png'))
                        model.save_image(cur_recons[idx].squeeze(), os.path.join(args.out_path, str(data_idx) + '_recon.png'))
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
    # truncate labels to match number of latents with potentially dropped last batch
    dataset.labels = dataset.labels[:latents.shape[0]]
    logging.info("\rEncoded %d batches with average losses (All: %.2f | MSE: %.2f | KL: %.2f) and %d latent vectors." % (batch_idx, avg_loss, avg_mse, avg_kl, latents.shape[0]))

    if args.export_latents:
        np.save(os.path.join(args.out_path, 'latents.npy'), latents)
        logging.info("Saved %d latent vectors to '%s'." % (latents.shape[0], os.path.join(args.out_path, 'latents.npy')))

    logging.info("Calculating similarities...")
    sims = calc_dists(latents)

    # parse precision ranks
    prec_ranks = [int(r) for r in args.ranks.split(',')]

    logging.info("Calculating metrics...")
    mean_latents, rel_sim_by_label, oth_sim_by_label, label_precision, label_counts = calc_metrics(latents, dataset.labels, sims, len(dataset.label_descs), prec_ranks, sim_metric='euclidean')
    for rank in prec_ranks:
        log_metrics(dataset.label_descs, rank, rel_sim_by_label, oth_sim_by_label, label_precision[rank], label_counts)

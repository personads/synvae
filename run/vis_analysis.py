import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json, logging

import numpy as np
import tensorflow as tf

from collections import defaultdict, OrderedDict

from data.cifar import Cifar
from utils.analysis import *
from utils.experiments import *
from models.visual import MnistVae, CifarVae


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='VisualVAE - Analysis')
    arg_parser.add_argument('task', choices=['mnist', 'cifar'], help='name of the task (mnist, cifar)')
    arg_parser.add_argument('model_path', help='path to VisualVAE model')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('data_split', choices=['train', 'test'], default='test', help='data split (train, test (default))')
    arg_parser.add_argument('out_path', help='path to output')
    arg_parser.add_argument('--beta', type=float, default=1., help='beta parameter for weighting KL-divergence (default: 1.0)')
    arg_parser.add_argument('--batch_size', type=int, default=200, help='batch size (default: 200)')
    arg_parser.add_argument('--ranks', default='1,5,10', help='precision ranks to use during evaluation (default: "1,5,10")')
    arg_parser.add_argument('--num_examples', default=4, help='number of examples for evaluation (default: 4)')
    arg_parser.add_argument('--num_tasks', default=20, help='number of tasks for evaluation (default: 20)')
    arg_parser.add_argument('--export', action='store_true', help='export original samples and reconstructions')
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
    elif args.task == 'cifar':
        model = CifarVae(latent_dim=512, beta=args.beta, batch_size=args.batch_size)
    model.build()

    # load data
    images, labels, label_descs, num_labels = load_data(args.task, split=args.data_split, data_path=args.data_path)

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
        
        reconstructions, latents = None, None
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
                if (reconstructions is None) or (latents is None):
                    reconstructions, latents = cur_recons, cur_latents
                else:
                    reconstructions = np.concatenate((reconstructions, cur_recons), axis=0)
                    latents = np.concatenate((latents, cur_latents), axis=0)
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
    logging.info("\rEncoded %d batches with average losses (All: %.2f | MSE: %.2f | KL: %.2f), %d reconstructions and %d latent vectors." % (batch_idx, avg_loss, avg_mse, avg_kl, reconstructions.shape[0], latents.shape[0]))

    if args.export:
        logging.info("Saving outputs...")
        for idx in range(images.shape[0]):
            model.save_image(images[idx].squeeze(), os.path.join(args.out_path, str(idx) + '_orig.png'))
            model.save_image(reconstructions[idx].squeeze(), os.path.join(args.out_path, str(idx) + '_recon.png'))
            sys.stdout.write("\rSaved %d/%d (%.2f%%)..." % (idx+1, images.shape[0], ((idx+1)*100)/images.shape[0]))
            sys.stdout.flush()
        logging.info("\rSaved %d images and reconstructions." % images.shape[0])

    logging.info("Calculating similarities...")
    sims = calc_dists(latents)

    # parse precision ranks
    prec_ranks = [int(r) for r in args.ranks.split(',')]

    logging.info("Calculating metrics...")
    mean_latents, rel_sim_by_label, oth_sim_by_label, label_precision = calc_metrics(latents, labels, sims, num_labels, prec_ranks, sim_metric='euclidean')
    for rank in prec_ranks:
        log_metrics(label_descs, rank, rel_sim_by_label, oth_sim_by_label, label_precision[rank])

    logging.info("Exporting evaluation samples...")
    examples, tasks = gen_eval_task(mean_latents, latents, labels, args.num_examples, args.num_tasks)
    eval_config = OrderedDict([
        ('name', args.task.upper()),
        ('code', ''),
        ('data_path', ''),
        ('result_path', ''),
        ('examples', examples),
        ('tasks', tasks)
    ])
    eval_config_path = os.path.join(args.out_path, 'eval.json')
    with open(eval_config_path, 'w', encoding='utf8') as fop:
        json.dump(eval_config, fop)
    logging.info("Saved evaluation configuration with %d examples and %d tasks to '%s'." % (len(examples), len(tasks), eval_config_path))

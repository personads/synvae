import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf

from data.cifar import Cifar
from models.visual import CifarVae

from experiments import *


if __name__ == '__main__':
    arg_parser = parse_arguments('CIFAR10 VAE')
    arg_parser.add_argument('cifar_path', help='path to CIFAR10')
    args = arg_parser.parse_args()
    model_path, tb_path, out_path = make_experiment_dir(args.exp_path)
    # set up model
    cifar_vae = CifarVae(latent_dim=512, batch_size=args.batch_size)
    # load CIFAR
    cifar = Cifar(args.cifar_path, verbose=True)
    train_images = cifar.data[:int(cifar.data.shape[0]*.8)]
    valid_images = cifar.data[int(cifar.data.shape[0]*.8):]
    print("Loaded %d training, %d validation images from CIFAR10." % (train_images.shape[0], valid_images.shape[0]))
    # set up TF datasets
    num_batches = train_images.shape[0] // args.batch_size + 1
    num_batches_valid = valid_images.shape[0] // args.batch_size + 1
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(args.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    train_next = train_iterator.get_next()
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_images).batch(args.batch_size)
    valid_iterator = valid_dataset.make_initializable_iterator()
    valid_next = valid_iterator.get_next()
    print("Loaded data into TensorFlow.")

    epochs = args.epochs
    with tf.Session() as sess:
        # initialize FileWriter for TensorBoard
        merge_summaries = tf.summary.merge_all()
        tf_writer = tf.summary.FileWriter(tb_path, graph=sess.graph)
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # epoch training loop
        min_loss = None
        for epoch in range(1, epochs + 1):
            sess.run(train_iterator.initializer)
            # iterate over batches
            avg_loss = 0.
            batch_idx = 0
            while True:
                batch_idx += 1
                try:
                    batch = sess.run(train_next)
                    _, cur_loss, summaries = sess.run([cifar_vae.train_op, cifar_vae.loss, merge_summaries], feed_dict={cifar_vae.images: batch})
                    avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                    sys.stdout.write("\repoch %d/%d. batch %d/%d. avg_loss %.2f. cur_loss %.2f.   " % (epoch, epochs, batch_idx, num_batches, avg_loss, cur_loss))
                    sys.stdout.flush()
                # end of dataset
                except tf.errors.OutOfRangeError:
                    # exit batch loop and proceed to next epoch
                    break
            # write epoch summary
            tf_writer.add_summary(summaries, epoch)
            print("\rcompleted epoch %d/%d. avg_loss %.2f.%s" % (epoch, epochs, avg_loss, ' '*32))

            # check performance on test split
            sess.run(valid_iterator.initializer)
            # iterate over batches
            avg_loss = 0.
            batch_idx = 0
            while True:
                batch_idx += 1
                try:
                    sys.stdout.write("\revaluating batch %d/%d." % (batch_idx, num_batches_valid))
                    sys.stdout.flush()
                    batch = sess.run(valid_next)
                    cur_loss, reconstructions = sess.run([cifar_vae.loss, cifar_vae.reconstructions], feed_dict={cifar_vae.images: batch})
                    avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                    # save original image and reconstruction
                    if (batch_idx-1) % 5 == 0:
                        if epoch == 1:
                            cifar_vae.save_image(batch[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(epoch) + '_orig.png'))
                        cifar_vae.save_image(reconstructions[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(epoch) + '_recon.png'))
                # end of dataset
                except tf.errors.OutOfRangeError:
                    # exit batch loop and proceed to next epoch
                    break
            print("\rcompleted evaluation. avg_loss %.2f." % avg_loss)

            # check if model has improved
            if (min_loss is None) or (avg_loss < min_loss):
                print("saving new best model with avg_loss %.2f." % avg_loss)
                cifar_vae.save(sess, os.path.join(model_path, 'best_model.ckpt'))
                min_loss = avg_loss
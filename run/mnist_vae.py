import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse

import numpy as np
import tensorflow as tf

from models.visual import MnistVae


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='MNIST VAE')
    arg_parser.add_argument('exp_path', help='path to experiment files (model checkpoints, TensorBoard logs, model outputs)')
    arg_parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    arg_parser.add_argument('--batch_size', type=int, default=100, help='batch size for training and evaluation')
    return arg_parser.parse_args()


def make_experiment_dir(path):
    # check if directory already exists
    if os.path.exists(path):
        print("[Error] '%s' already exists." % (path,))
        sys.exit()
    # make necessary directories
    os.mkdir(path)
    checkpoints_path = os.path.join(path, 'checkpoints')
    os.mkdir(checkpoints_path)
    tensorboard_path = os.path.join(path, 'tensorboard')
    os.mkdir(tensorboard_path)
    output_path = os.path.join(path, 'output')
    os.mkdir(output_path)
    return checkpoints_path, tensorboard_path, output_path


if __name__ == '__main__':
    args = parse_arguments()
    model_path, tb_path, out_path = make_experiment_dir(args.exp_path)
    # set up model
    mnist_vae = MnistVae(latent_dim=50, batch_size=args.batch_size)
    # load MNIST
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    print("Loaded %d training, %d test images from MNIST." % (train_images.shape[0], test_images.shape[0]))
    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.
    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.
    # set up TF datasets
    num_batches = train_images.shape[0] // args.batch_size + 1
    num_batches_test = test_images.shape[0] // args.batch_size + 1
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(args.batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    train_next = train_iterator.get_next()
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(args.batch_size)
    test_iterator = test_dataset.make_initializable_iterator()
    test_next = test_iterator.get_next()

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
                    _, cur_loss, summaries = sess.run([mnist_vae.train_op, mnist_vae.loss, merge_summaries], feed_dict={mnist_vae.images: batch})
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
            sess.run(test_iterator.initializer)
            # iterate over batches
            avg_loss = 0.
            batch_idx = 0
            while True:
                batch_idx += 1
                try:
                    sys.stdout.write("\revaluating batch %d/%d." % (batch_idx, num_batches_test))
                    sys.stdout.flush()
                    batch = sess.run(test_next)
                    cur_loss, reconstructions = sess.run([mnist_vae.loss, mnist_vae.reconstructions], feed_dict={mnist_vae.images: batch})
                    avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                    # save original image and reconstruction
                    if (batch_idx-1) % 5 == 0:
                        if epoch == 1:
                            mnist_vae.save_image(batch[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(epoch) + '_orig.png'))
                        mnist_vae.save_image(reconstructions[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(epoch) + '_recon.png'))
                # end of dataset
                except tf.errors.OutOfRangeError:
                    # exit batch loop and proceed to next epoch
                    break
            print("\rcompleted evaluation. avg_loss %.2f." % avg_loss)

            # check if model has improved
            if (min_loss is None) or (avg_loss < min_loss):
                print("saving new best model with avg_loss %.2f." % avg_loss)
                mnist_vae.save(sess, os.path.join(model_path, 'best_model.ckpt'))
                min_loss = avg_loss
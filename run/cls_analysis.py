import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import tensorflow as tf

from utils.analysis import *
from utils.experiments import *
from models.classifiers import MnistCnn, CifarCnn


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='VisualCNN - Analysis')
    arg_parser.add_argument('task', choices=['mnist', 'cifar'], help='name of the task (mnist, cifar)')
    arg_parser.add_argument('model_path', help='path to VisualCNN model')
    arg_parser.add_argument('data_path', help='path to data (not required for original MNIST)')
    arg_parser.add_argument('out_path', help='path to output')
    arg_parser.add_argument('--batch_size', type=int, default=200, help='batch size (default: 200)')
    args = arg_parser.parse_args()

    # check if directory already exists
    if os.path.exists(args.out_path):
        print("[Error] '%s' already exists." % (args.out_path,))
        sys.exit()
    # make necessary directories
    os.mkdir(args.out_path)

    setup_logging(os.path.join(args.out_path, 'results.log'))

    # set up classifier
    if args.task == 'mnist':
        model = MnistCnn(batch_size=args.batch_size)
    elif args.task == 'cifar':
        model = CifarCnn(batch_size=args.batch_size)
    model.build()

    # load data
    images, labels, label_descs, num_labels = load_data(args.task, split='test', data_path=args.data_path)

    # set up TF datasets
    dataset = tf.data.Dataset.from_tensor_slices({'images': images, 'labels': labels}).batch(args.batch_size)
    iterator = dataset.make_initializable_iterator()
    next_op = iterator.get_next()

    # inference
    with tf.Session() as sess:
        # initialize variables and dataset iterator
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        # restore MusicVAE
        model.restore(tf_session=sess, path=args.model_path)

        # iterate over batches
        predictions = None
        avg_loss = 0.
        batch_idx = 0
        while True:
            try:
                sys.stdout.write("\rClassifying batch %d..." % (batch_idx))
                sys.stdout.flush()
                batch = sess.run(next_op)
                batch_idx += 1
                cur_loss, cur_predictions = sess.run([model.loss, model.predictions], feed_dict={model.images: batch['images'], model.labels: batch['labels']})
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                # append to result
                if predictions is None:
                    predictions = cur_predictions
                else:
                    predictions = np.concatenate((predictions, cur_predictions), axis=0)
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
    logging.info("\rClassified %d images with avg_loss %.2f." % (predictions.shape[0], avg_loss))

    logging.info("Calculating metrics...")
    total_precision, label_precision, label_recall, label_accuracy = calc_cls_metrics(labels, predictions)

    logging.info("Metrics by class:")
    for label_idx, label in enumerate(label_descs):
        logging.info("  %s: %.2f Accuracy, %.2f Precision, %.2f Recall." % (
            label, label_accuracy[label_idx],
            label_precision[label_idx],
            label_recall[label_idx]
            ))
    logging.info("Total Precision: %.2f." % total_precision)

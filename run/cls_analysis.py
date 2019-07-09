import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

import tensorflow as tf

from data import *
from models.classifiers import *
from utils.analysis import *
from utils.experiments import *


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='VisualCNN - Analysis')
    arg_parser.add_argument('task', choices=['mnist', 'cifar', 'bam'], help='name of the task')
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
        dataset = Mnist(split='test', data_path=args.data_path)
    elif args.task == 'cifar':
        model = CifarCnn(batch_size=args.batch_size)
        dataset = Cifar(args.data_path)
    elif args.task == 'bam':
        dataset = Bam(args.data_path)
        dataset.filter_labels(['emotion_gloomy', 'emotion_happy', 'emotion_peaceful', 'emotion_scary'])
        dataset.filter_uncertain(round_up=True)
        model = BamCnn(num_labels=len(dataset.label_descs), batch_size=args.batch_size)
    model.build()

    # load data
    iterator = dataset.get_iterator(batch_size=args.batch_size)
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
    dataset.labels = dataset.labels[:predictions.shape[0]]
    logging.info("\rClassified %d images with avg_loss %.2f." % (predictions.shape[0], avg_loss))

    logging.info("Calculating metrics...")
    if args.task == 'bam':
        avg_accuracy, label_precision, label_recall, label_accuracy = calc_mltcls_metrics(dataset.labels, predictions)
    else:
        avg_accuracy, label_precision, label_recall, label_accuracy = calc_cls_metrics(dataset.labels, predictions)

    logging.info("Metrics by class:")
    for label_idx, label in enumerate(dataset.label_descs):
        logging.info("  %s: %.2f Accuracy, %.2f Precision, %.2f Recall." % (
            label, label_accuracy[label_idx],
            label_precision[label_idx],
            label_recall[label_idx]
            ))
    logging.info("Mean Accuracy: %.2f." % avg_accuracy)

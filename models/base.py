import logging, os, sys

import tensorflow as tf

from collections import defaultdict

class BaseModel:
    def __init__():
        pass


    def save(self, tf_session, path, var_list=None):
        save_path = tf.train.Saver(var_list=None).save(tf_session, path)
        logging.info("[%s] Saved model to '%s'." % (self.__class__.__name__, save_path))


    def restore(self, tf_session, path, var_list=None):
        tf.train.Saver(var_list).restore(tf_session, path)
        logging.info("[%s] Restored model from '%s'." % (self.__class__.__name__, path))


    def run_train_step(self, tf_session, batch):
        pass


    def run_test_step(self, tf_session, batch, batch_idx, out_path):
        pass


    def train(self, tf_session, train_iter, valid_iter, max_epochs, model_path, out_path, tf_writer):
        # set up training specific ops
        self.merge_op = tf.summary.merge_all()
        next_op = train_iter.get_next()
        valid_next_op = valid_iter.get_next()
        # epoch training loop
        min_loss = None
        while self.epoch < max_epochs:
            self.epoch += 1
            tf_session.run(train_iter.initializer)
            # iterate over batches
            avg_losses = defaultdict(float)
            batch_idx = 0
            while True:
                try:
                    batch = tf_session.run(next_op)
                    batch_idx += 1
                    cur_losses, cur_summaries = self.run_train_step(tf_session, batch)
                    for loss in cur_losses:
                        avg_losses[loss] = ((avg_losses[loss] * (batch_idx - 1)) + cur_losses[loss]) / batch_idx
                    avg_losses_str = ' | '.join(['%s: %.2f' % (l, avg_losses[l]) for l in sorted(avg_losses)])
                    cur_losses_str = ' | '.join(['%s: %.2f' % (l, cur_losses[l]) for l in sorted(cur_losses)])
                    sys.stdout.write("\rEpoch %d/%d. Batch %d. Average losses (%s). Current losses (%s).   " % (self.epoch, max_epochs, batch_idx, avg_losses_str, cur_losses_str))
                    sys.stdout.flush()
                # end of dataset
                except tf.errors.OutOfRangeError:
                    # exit batch loop and proceed to next epoch
                    break
            # write epoch summary
            tf_writer.add_summary(cur_summaries, self.epoch)
            logging.info("\r[%s] Completed epoch %d/%d (%d batches). Average losses (%s).%s" % (self.__class__.__name__, self.epoch, max_epochs, batch_idx, avg_losses_str, ' '*len(cur_losses_str)))

            # check performance on test split
            valid_losses = self.test(tf_session, valid_iter, valid_next_op, out_path)
            valid_losses_str = ' | '.join(['%s: %.2f' % (l, valid_losses[l]) for l in sorted(valid_losses)])
           
            # save latest model
            logging.info("Saving latest model...")
            self.save(tf_session, os.path.join(model_path, 'latest_model.ckpt'))
            # check if model has improved
            if (min_loss is None) or (valid_losses['All'] < min_loss):
                logging.info("Saving best model with validation losses (%s)..." % valid_losses_str)
                self.save(tf_session, os.path.join(model_path, 'best_model.ckpt'))
                min_loss = valid_losses['All']


    def test(self, tf_session, iterator, next_op, out_path):
        tf_session.run(iterator.initializer)
        # iterate over batches
        avg_losses = defaultdict(float)
        batch_idx = 0
        while True:
            try:
                sys.stdout.write("\rEvaluating batch %d..." % (batch_idx))
                sys.stdout.flush()
                batch = tf_session.run(next_op)
                batch_idx += 1
                cur_losses = self.run_test_step(tf_session, batch, batch_idx, out_path)
                for loss in cur_losses:
                    avg_losses[loss] = ((avg_losses[loss] * (batch_idx - 1)) + cur_losses[loss]) / batch_idx
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
        avg_losses_str = ' | '.join(['%s: %.2f' % (l, avg_losses[l]) for l in sorted(avg_losses)])
        logging.info("\r[%s] Completed evaluation (%d batches). Average losses (%s)." % (self.__class__.__name__, batch_idx, avg_losses_str))
        return avg_losses

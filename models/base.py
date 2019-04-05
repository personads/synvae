import logging, os, sys

import tensorflow as tf

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
            avg_loss = 0.
            batch_idx = 0
            while True:
                try:
                    batch = tf_session.run(next_op)
                    batch_idx += 1
                    cur_loss, cur_summaries = self.run_train_step(tf_session, batch)
                    avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                    sys.stdout.write("\rEpoch %d/%d. Batch %d. avg_loss %.2f. cur_loss %.2f.   " % (self.epoch, max_epochs, batch_idx, avg_loss, cur_loss))
                    sys.stdout.flush()
                # end of dataset
                except tf.errors.OutOfRangeError:
                    # exit batch loop and proceed to next epoch
                    break
            # write epoch summary
            tf_writer.add_summary(cur_summaries, self.epoch)
            logging.info("\r[%s] Completed epoch %d/%d (%d batches). avg_loss %.2f.%s" % (self.__class__.__name__, self.epoch, max_epochs, batch_idx, avg_loss, ' '*32))

            # check performance on test split
            valid_loss = self.test(tf_session, valid_iter, valid_next_op, out_path)
           
            # save latest model
            logging.info("Saving latest model...")
            self.save(tf_session, os.path.join(model_path, 'latest_model.ckpt'))
            # check if model has improved
            if (min_loss is None) or (valid_loss < min_loss):
                logging.info("Saving best model with valid_loss %.2f..." % valid_loss)
                self.save(tf_session, os.path.join(model_path, 'best_model.ckpt'))
                min_loss = valid_loss


    def test(self, tf_session, iterator, next_op, out_path):
        tf_session.run(iterator.initializer)
        # iterate over batches
        avg_loss = 0.
        batch_idx = 0
        while True:
            try:
                sys.stdout.write("\rEvaluating batch %d..." % (batch_idx))
                sys.stdout.flush()
                batch = tf_session.run(next_op)
                batch_idx += 1
                temperature = 0.5
                cur_loss = self.run_test_step(tf_session, batch, batch_idx, out_path)
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
        logging.info("\r[%s] Completed evaluation (%d batches). avg_loss %.2f." % (self.__class__.__name__, batch_idx, avg_loss))
        return avg_loss

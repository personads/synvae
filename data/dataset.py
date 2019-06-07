import logging

import tensorflow as tf

class Dataset:
    '''Dataloader Superclass'''
    def __init__(self, split, data_path=None):
        self.data, self.label_descs = None, None


    def split_train_data(self):
        split_idx = int(self.data.shape[0]*.8)
        train_images, train_labels = self.data[:split_idx], self.labels[:split_idx]
        valid_images, valid_labels = self.data[split_idx:], self.labels[split_idx:]
        logging.info("[%s] Split data into %d training and %d validation images." % (self.__class__.__name__, train_images.shape[0], valid_images.shape[0]))
        return train_images, train_labels, valid_images, valid_labels


    def get_iterator(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices({'images': self.data, 'labels': self.labels}).batch(batch_size, drop_remainder=True)
        iterator = dataset.make_initializable_iterator()
        return iterator


    def get_image_iterator(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(self.data).batch(batch_size, drop_remainder=True)
        iterator = dataset.make_initializable_iterator()
        return iterator


    def get_train_iterators(self, batch_size):
        train_images, train_labels, valid_images, valid_labels = self.split_train_data()
        train_dataset = tf.data.Dataset.from_tensor_slices({'images': train_images, 'labels': train_labels}).shuffle(train_images.shape[0]).batch(batch_size, drop_remainder=True)
        train_iterator = train_dataset.make_initializable_iterator()
        valid_dataset = tf.data.Dataset.from_tensor_slices({'images': valid_images, 'labels': valid_labels}).batch(batch_size, drop_remainder=True)
        valid_iterator = valid_dataset.make_initializable_iterator()
        return train_iterator, valid_iterator


    def get_train_image_iterators(self, batch_size, buffer_size=1000):
        train_images, _, valid_images, _ = self.split_train_data()
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        train_iterator = train_dataset.make_initializable_iterator()
        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_images).batch(batch_size, drop_remainder=True)
        valid_iterator = valid_dataset.make_initializable_iterator()
        return train_iterator, valid_iterator

import logging, os

import tensorflow as tf

from data.dataset import Dataset

class Mnist(Dataset):
    '''MNIST Dataloader'''
    def __init__(self, split, data_path=None):
        # load MNIST
        if (data_path is None) or (len(data_path) < 1):
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
            if split == 'train':
                images, labels = train_images, train_labels
            elif split == 'test':
                images, labels = test_images, test_labels
        else:
            with open(data_path, 'rb') as fop:
                images, labels = pickle.load(fop)
        images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
        # normalizing the images to the range of [0., 1.]
        images /= 255.
        # binarization
        images[images >= .5] = 1.
        images[images < .5] = 0.
        # init internal variables
        self.data, self.labels = images, labels
        self.label_descs = [str(i) for i in range(10)]
        logging.info("[MNIST] Loaded %d %s images." % (self.data.shape[0], split))


    def split_train_data(self):
        split_idx = 50000
        train_images, train_labels = self.data[:split_idx], self.labels[:split_idx]
        valid_images, valid_labels = self.data[split_idx:], self.labels[split_idx:]
        logging.info("[MNIST] Split data into %d training and %d validation images." % (train_images.shape[0], valid_images.shape[0]))
        return train_images, train_labels, valid_images, valid_labels

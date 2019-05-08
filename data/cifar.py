import logging, os, pickle

import numpy as np

from data.dataset import Dataset

class Cifar(Dataset):
    '''CIFAR Dataloader'''
    def __init__(self, cifar_dir):
        self.data, self.labels = self._load_cifar_directory(cifar_dir)
        self.label_descs = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


    def _load_cifar_directory(self, cifar_dir):
        data, labels = None, None
        paths = os.listdir(cifar_dir)
        for path_idx, path in enumerate(paths):
            if not path.endswith('.pkl'):
                continue
            pickle_path = os.path.join(cifar_dir, path)
            cur_data, cur_labels = self._load_cifar_pickle(pickle_path)
            # append to result
            if (data is None) or (labels is None):
                data, labels = cur_data, cur_labels
            else:
                data = np.concatenate((data, cur_data), axis=0)
                labels = np.concatenate((labels, cur_labels), axis=0)
        return data, labels


    def _load_cifar_pickle(self, pickle_path):
        '''Loads a CIFAR10 pickle.

        Returns:
            data (np.array): [num_data, height, width, RGB]
            labels (np.array): [num_data]
        '''
        with open(pickle_path, 'rb') as fo:
            cifar_dict = pickle.load(fo, encoding='bytes')
            data, labels = cifar_dict[b'data'], cifar_dict[b'labels']
        # reshape images
        data = data.reshape([-1, 3, 32, 32]) # num_data, RGB, 32x32
        data = data.transpose([0, 2, 3, 1]) # num_data, height, width, RGB
        data = data.astype(float) # convert to float
        data /= 255. # normalize values to [0., 1.]
        labels = np.array(labels, dtype=int) # convert labels to numpy array
        logging.info("[CIFAR] Loaded %d images from '%s'." % (cifar_dict[b'data'].shape[0], pickle_path))
        return data, labels


    def split_train_data(self):
        split_idx = int(self.data.shape[0]*.8)
        train_images, train_labels = self.data[:split_idx], self.labels[:split_idx]
        valid_images, valid_labels = self.data[split_idx:], self.labels[split_idx:]
        logging.info("[CIFAR] Split data into %d training and %d validation images." % (train_images.shape[0], valid_images.shape[0]))
        return train_images, train_labels, valid_images, valid_labels

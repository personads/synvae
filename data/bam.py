import logging, multiprocessing, os

import numpy as np
import tensorflow as tf

from data.dataset import Dataset

class Bam(Dataset):
    '''BAM Dataloader'''
    def __init__(self, data_path):
        # init internal variables
        self.data = [os.path.join(data_path, 'img/', cur_path) for cur_path in os.listdir(os.path.join(data_path, 'img/'))]
        self.labels = np.load(os.path.join(data_path, 'labels.npy'))
        self.label_descs = [
            'content_bicycle', 'content_bird', 'content_building', 'content_cars', 'content_cat', 'content_dog', 'content_flower', 'content_people', 'content_tree',
            'emotion_gloomy', 'emotion_happy', 'emotion_peaceful', 'emotion_scary',
            'media_oilpaint', 'media_watercolor']
        logging.info("[BAM] Found %d images in '%s'." % (len(self.data), data_path))


    def _load_image(path):
        image = tf.read_file(path)
        image = tf.cast(tf.image.decode_jpeg(image, channels=3), dtype=tf.float32)
        image /= 255.0
        return image


    def split_train_data(self):
        split_idx = int(len(self.data)*.8)
        train_images, train_labels = self.data[:split_idx], self.labels[:split_idx]
        valid_images, valid_labels = self.data[split_idx:], self.labels[split_idx:]
        logging.info("[MNIST] Split data into %d training and %d validation images." % (train_images.shape[0], valid_images.shape[0]))
        return train_images, train_labels, valid_images, valid_labels


    def get_train_image_iterators(self, batch_size, buffer_size=3*1e4):
        train_images, _, valid_images, _ = self.split_train_data()
        # construct training dataset
        train_paths = tf.data.Dataset.from_tensor_slices(train_images)
        train_dataset = train_paths.map(self._load_image, num_parallel_calls=multiprocessing.cpu_count())
        train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
        train_iterator = train_dataset.make_initializable_iterator()
        # construct validation dataset
        valid_paths = tf.data.Dataset.from_tensor_slices(valid_images)
        valid_dataset = valid_paths.map(self._load_image, num_parallel_calls=multiprocessing.cpu_count())
        valid_dataset = valid_dataset.batch(batch_size)
        valid_iterator = valid_dataset.make_initializable_iterator()
        return train_iterator, valid_iterator

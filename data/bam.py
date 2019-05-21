import logging, multiprocessing, os

import numpy as np
import tensorflow as tf

from data.dataset import Dataset

class Bam(Dataset):
    '''BAM Dataloader'''
    def __init__(self, data_path):
        # init internal variables
        self.data = [os.path.join(data_path, 'img/', cur_path) for cur_path in os.listdir(os.path.join(data_path, 'img/'))]
        self.data.sort(key=lambda el: int(os.path.splitext(os.path.basename(el))[0])) #sort by MID
        self.labels = np.load(os.path.join(data_path, 'labels.npy'))
        self.label_descs = [
            'content_bicycle', 'content_bird', 'content_building', 'content_cars', 'content_cat', 'content_dog', 'content_flower', 'content_people', 'content_tree',
            'emotion_gloomy', 'emotion_happy', 'emotion_peaceful', 'emotion_scary',
            'media_oilpaint', 'media_watercolor']
        logging.info("[BAM] Found %d images in '%s'." % (len(self.data), data_path))


    def _load_train_image(self, path):
        image = tf.read_file(path)
        image = tf.cast(tf.image.decode_jpeg(image, channels=3), dtype=tf.float32)
        image = tf.image.random_crop(image, [256, 256, 3])
        image /= 255.0
        return image


    def _load_test_image(self, path):
        image = tf.read_file(path)
        image = tf.cast(tf.image.decode_jpeg(image, channels=3), dtype=tf.float32)
        # crop to centre
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        min_size = tf.minimum(height, width)
        image = tf.image.crop_to_bounding_box(image, (height - min_size)//2, (width - min_size)//2, 256, 256)
        image /= 255.0
        return image


    def split_train_data(self):
        split_idx = int(len(self.data)*.8)
        train_images, train_labels = self.data[:split_idx], self.labels[:split_idx]
        valid_images, valid_labels = self.data[split_idx:], self.labels[split_idx:]
        logging.info("[BAM] Split data into %d training and %d validation images." % (len(train_images), len(valid_images)))
        return train_images, train_labels, valid_images, valid_labels


    def get_image_iterator(self, batch_size):
        paths = tf.data.Dataset.from_tensor_slices(self.data)
        dataset = paths.map(self._load_test_image, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(batch_size, drop_remainder=True)
        iterator = dataset.make_initializable_iterator()
        return iterator


    def get_train_image_iterators(self, batch_size, buffer_size=1000):
        train_images, _, valid_images, _ = self.split_train_data()
        # construct training dataset
        train_paths = tf.data.Dataset.from_tensor_slices(train_images)
        train_dataset = train_paths.map(self._load_train_image, num_parallel_calls=multiprocessing.cpu_count())
        train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        train_iterator = train_dataset.make_initializable_iterator()
        # construct validation dataset
        valid_paths = tf.data.Dataset.from_tensor_slices(valid_images)
        valid_dataset = valid_paths.map(self._load_test_image, num_parallel_calls=multiprocessing.cpu_count())
        valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)
        valid_iterator = valid_dataset.make_initializable_iterator()
        return train_iterator, valid_iterator

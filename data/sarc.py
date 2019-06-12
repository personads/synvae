import json, logging, multiprocessing, os

import numpy as np
import tensorflow as tf

from data.dataset import Dataset

class Sarc(Dataset):
    '''Sarc Dataloader'''
    def __init__(self, data_path):
        # init internal variables
        self.data = np.load(os.path.join(data_path, 'data.npy'))
        self.labels = np.load(os.path.join(data_path, 'labels.npy'))
        self.idx_tkn_map, self.tkn_idx_map = self._load_maps(data_path)
        self.max_length = self.data.shape[1]
        self.vocab_size = len(self.idx_tkn_map)
        self.label_descs = ['non-sarcastic', 'sarcastic']
        logging.info("[SARC] Loaded %d text blocks with %d unique tokens and maximum length %d from '%s'." % (self.data.shape[0], self.vocab_size, self.max_length, data_path))


    def _load_maps(self, data_path):
        idx_tkn_map = json.load(open(os.path.join(data_path, 'idx_tkn.json'), 'r', encoding='utf8'))
        tkn_idx_map = {c: i for i, c in enumerate(idx_tkn_map)}
        return idx_tkn_map, tkn_idx_map

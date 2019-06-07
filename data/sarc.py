import json, logging, multiprocessing, os

import numpy as np
import tensorflow as tf

from data.dataset import Dataset

class Sarc(Dataset):
    '''Sarc Dataloader'''
    def __init__(self, data_path, max_length=100):
        # init internal variables
        self.txt_data = self._load_data(os.path.join(data_path, 'data.txt'))[:1000]
        self.idx_tkn_map, self.tkn_idx_map = self._load_maps(data_path)
        self.max_length = max_length
        self.vocab_size = len(self.idx_tkn_map)
        self.data = self._convert_texts_to_arrays(self.txt_data)
        self.labels = np.load(os.path.join(data_path, 'labels.npy'))
        self.label_descs = ['non-sarcastic', 'sarcastic']
        logging.info("[SARC] Loaded %d text blocks with %d unique tokens and maximum length %d from '%s'." % (self.data.shape[0], self.vocab_size, self.max_length, data_path))


    def _load_data(self, text_path):
        data = []
        with open(text_path, 'r', encoding='utf8') as fop:
            for line in fop:
                if len(line) < 1:
                    continue
                data.append(line.strip().split())
        return data


    def _load_maps(self, data_path):
        idx_tkn_map = ['<pad>'] + json.load(open(os.path.join(data_path, 'idx_tkn.json'), 'r', encoding='utf8')) + ['<unk>', '<s>', '</s>']
        tkn_idx_map = {c: i for i, c in enumerate(idx_tkn_map)}
        return idx_tkn_map, tkn_idx_map


    def _convert_texts_to_arrays(self, texts, padding=True):
        res = None
        for text in texts:
            if len(text) > self.max_length - 3:
                continue
            arr_text = [self.tkn_idx_map['<s>']] + [self.tkn_idx_map.get(t, self.tkn_idx_map['<unk>']) for t in text] + [self.tkn_idx_map['</s>']]
            if padding and (len(arr_text) < self.max_length):
                arr_text += [0 for _ in range(self.max_length - len(arr_text))]
            arr_text = np.array(arr_text, dtype=int).reshape([1, self.max_length])
            if res is None:
                res = arr_text
            else:
                res = np.concatenate((res, arr_text), axis=0)
        return res

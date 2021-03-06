import logging, os, pickle

import numpy as np

from data.dataset import Dataset

class Latents(Dataset):
    '''Latents Dataloader'''
    def __init__(self, vis_path, aud_path):
        self.data = self._load_latents(vis_path, aud_path)
        self.labels = np.zeros([self.data.shape[0]])
        self.label_desc = 'null'
        self.latent_dim = self.data.shape[1]//2
        logging.info("[Latents] Loaded %d visual and auditive latent vectors." % (self.data.shape[0],))


    def _load_latents(self, vis_path, aud_path):
        vis_latents = np.load(vis_path)
        aud_latents = np.load(aud_path)
        # truncate to common size
        min_size = min(vis_latents.shape[0], aud_latents.shape[0])
        return np.concatenate((vis_latents[:min_size], aud_latents[:min_size]), axis=1)

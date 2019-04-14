import json
import numpy as np
from tqdm import tqdm

from construct_dataset import num_type
from util import load_dict

class TypedataLoader():
    def __init__(self, data_file, entity2id):
        self.entity2id = entity2id
        print('loading data from', data_file)
        self.data = []
        with open(data_file) as f_in:
            for line in tqdm(f_in):
                line = json.loads(line)
                self.data.append(line)
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)
        self.entities = np.full((self.num_data, 1), len(self.entity2id), dtype=int)
        self.answer_dist = np.zeros((self.num_data), dtype=int)

        self._prepare_data()
    
    def _prepare_data(self):
        next_id = 0
        for sample in tqdm(self.data):
            self.entities[next_id, 0] = self.entity2id[list(sample.keys())[0]]
            self.answer_dist[next_id] = int(list(sample.values())[0])
            next_id += 1
    
    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)
    
    def get_batch(self, iteration, batch_size):
        sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
        return self.entities[sample_ids], self.answer_dist[sample_ids]
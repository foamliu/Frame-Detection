import pickle

import numpy as np

from config import im_size

if __name__ == "__main__":
    with open('data/data.pkl', 'rb') as file:
        data = pickle.load(file)

    samples = [item for item in data if not item['is_sample'] and item['pts']]

    for sample in samples:
        pts = sample['pts']
        pts = np.array(sample['pts'])
        pts = pts.reshape((8,))

        minimal = np.min(pts)
        maximal = np.max(pts)
        if minimal < 0 or maximal >= im_size:
            print(sample)

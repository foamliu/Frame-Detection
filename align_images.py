import pickle
import random

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from config import im_size, pickle_file, num_train
from data_gen import data_transforms
from utils import ensure_folder, draw_bboxes

if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = [item for item in data if not item['is_sample'] and item['pts']]
    print([samples[0]])
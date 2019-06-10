import pickle

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from config import im_size, device, pickle_file, num_train
from data_gen import data_transforms

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = [item for item in data if not item['is_sample'] and item['pts']]
    samples = samples[num_train:]
    samples = np.random.sample(samples, 10)

    imgs = torch.zeros([10, 3, im_size, im_size], dtype=torch.float)

    for i in range(10):
        sample = samples[i]
        fullpath = sample['fullpath']
        img = cv.imread(fullpath)
        img = cv.resize(img, (im_size, im_size))
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        imgs[i] = img

    outputs = model(imgs)

    for i in range(10):
        output = outputs[i].cpu().numpy()
        print('output: ' + str(output))
        print('output.shape: ' + str(output.shape))

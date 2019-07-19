import pickle
import random

import cv2 as cv
import numpy as np
import torch
from models import FrameDetectionModel
from torchvision import transforms

from config import im_size, device, pickle_file, num_train
from data_gen import data_transforms
from utils import ensure_folder, draw_bboxes

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = FrameDetectionModel().load_state_dict(checkpoint['model'].state_dict())
    model = model.to(torch.device('cpu'))
    model.eval()

    transformer = data_transforms['valid']

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = [item for item in data if not item['is_sample'] and item['pts']]
    samples = samples[num_train:]
    samples = random.sample(samples, 10)

    imgs = torch.zeros([10, 3, im_size, im_size], dtype=torch.float)
    ensure_folder('images')

    for i in range(10):
        sample = samples[i]
        fullpath = sample['fullpath']
        raw = cv.imread(fullpath)
        raw = cv.resize(raw, (im_size, im_size))
        img = raw[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        imgs[i] = img

        cv.imwrite('images/{}_img.jpg'.format(i), raw)
        raw = draw_bboxes(raw, sample['pts'])
        cv.imwrite('images/{}_true.jpg'.format(i), raw)

    with torch.no_grad():
        outputs = model(imgs)

    for i in range(10):
        output = outputs[i].cpu().numpy()
        output = np.reshape(output, (4, 1, 2))
        output = output * im_size
        print('output: ' + str(output))
        print('output.shape: ' + str(output.shape))

        img = cv.imread('images/{}_img.jpg'.format(i))
        img = draw_bboxes(img, output)
        cv.imwrite('images/{}_out.jpg'.format(i), img)

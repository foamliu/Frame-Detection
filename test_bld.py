import os

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from config import im_size, device
from data_gen import data_transforms
from utils import ensure_folder, draw_bboxes

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    file_list = ['test_bld_1.jpg', 'test_bld_2.jpg', 'test_bld_3.jpg', 'test_bld_4.jpg']

    imgs = torch.zeros([4, 3, im_size, im_size], dtype=torch.float)
    ensure_folder('images')

    for i in range(len(file_list)):
        file = file_list[i]
        fullpath = os.path.join('images', file)
        print(fullpath)
        img = cv.imread(fullpath)
        img = cv.resize(img, (im_size, im_size))
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        imgs[i] = img

    with torch.no_grad():
        outputs = model(imgs)

    for i in range(len(file_list)):
        output = outputs[i].cpu().numpy()
        output = np.reshape(output, (4, 1, 2))
        output = output * im_size
        print('output: ' + str(output))
        print('output.shape: ' + str(output.shape))

        img = cv.imread('images/test_bld_{}.jpg'.format(i))
        img = draw_bboxes(img, output)
        cv.imwrite('images/out_bld_{}.jpg'.format(i), img)

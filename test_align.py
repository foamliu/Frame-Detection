import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from config import im_size
from data_gen import data_transforms
from utils import draw_bboxes

if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module
    model = model.to(torch.device('cpu'))
    model.eval()

    transformer = data_transforms['valid']

    fullpath = 'test/1.jpg'
    raw = cv.imread(fullpath)
    h, w = raw.shape[:2]
    img = cv.resize(raw, (im_size, im_size))
    img = img[..., ::-1]  # RGB
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

    output = outputs[0].cpu().numpy()
    output = np.reshape(output, (4, 1, 2))

    for p in output:
        p[0][0] = p[0][0] * w
        p[0][1] = p[0][0] * h

    # output = output * im_size
    print('output: ' + str(output))
    print('output.shape: ' + str(output.shape))

    img = draw_bboxes(raw, output)
    cv.imwrite('test/result.jpg', img)

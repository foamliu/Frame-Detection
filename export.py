import time
from mobilefacenet import MobileFaceNet
import torch

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model'].module
    print(type(model))

    filename = 'corner_detector.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(filename))
    start = time.time()
    model = MobileFaceNet()
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))

    scripted_model_file = 'corner_detector_scripted.pt'
    print('saving {}...'.format(scripted_model_file))
    torch.jit.save(torch.jit.script(model), scripted_model_file)

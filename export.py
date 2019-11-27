import time
from models import FrameDetectionModel
import torch

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['model'].module
    print(type(model))

    filename = 'framedetector.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model.state_dict(), filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(filename))
    start = time.time()
    model = FrameDetectionModel()
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))

    scripted_model_file = 'framedetector_scripted.pt'
    print('saving {}...'.format(scripted_model_file))
    torch.jit.save(torch.jit.script(model), scripted_model_file)

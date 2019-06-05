import json
import os
import pickle

from tqdm import tqdm

from config import DATA_DIR, IMG_DIR
from utils import ensure_folder, do_match

if __name__ == "__main__":
    ensure_folder(DATA_DIR)
    data = []
    dir_list = [d for d in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, d))]
    for dir in tqdm(dir_list):
        dir_path = os.path.join(IMG_DIR, dir)
        file_list = [f for f in os.listdir(dir_path) if f.lower().endswith('.jpg')]
        for file in file_list:
            fullpath = os.path.join(dir_path, file)
            is_sample = file == '0.jpg'
            data.append({'fullpath': fullpath, 'file': file, 'dir': dir, 'is_sample': is_sample})
    with open('data/file_list.json', 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    file_count = len(data)
    print('file_count: ' + str(file_count))

    not_sample = [item for item in data if not item['is_sample']]
    print('len(not_sample): ' + str(len(not_sample)))

    for item in not_sample:
        fullpath = item['fullpath']
        file = item['file']
        sample = fullpath.replace(file, '0.jpg')
        # print('fullpath: ' + fullpath)
        # print('sample: ' + sample)
        dst = do_match(sample, fullpath)
        item['pts'] = dst.tolist()

    with open('data/data.pkl', 'wb') as file:
        pickle.dump(data, file)

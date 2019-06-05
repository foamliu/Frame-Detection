import json
import os

from tqdm import tqdm

from config import DATA_DIR, IMG_DIR
from utils import ensure_folder

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

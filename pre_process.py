import pickle
from config import pickle_file

if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    print(len(data))
    print(data[0])
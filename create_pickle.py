import argparse
import glob
import pickle

from os import listdir, makedirs
from os.path import join

def pickle_data(file, data_dict):
    with open(file, 'wb') as fo:
        pickle.dump(data_dict, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='bytes')
    return _dict

def main(dataset_path, path):
    makedirs(path, exist_ok=True)

    out_path = join(path, 'mypickle.pickle')   
    classes_names = sorted(listdir(dataset_path))

    mypickle = {"Filenames": [], "Labels": []}

    for index, classname in enumerate(classes_names):
        for image in glob.glob(join(dataset_path, classname, '*.*')):
            mypickle["Filenames"].append(image)
            mypickle["Labels"].append(index)

    pickle_data(out_path, mypickle)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", help="set dataset folder path")
    parser.add_argument("--output_path", "-o", default='../data/', help="set output path")

    args = parser.parse_args()

    main(args.dataset_path, args.output_path)

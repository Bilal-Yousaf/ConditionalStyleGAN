import argparse
import glob
import pickle

import numpy as np
from PIL import Image

from os import listdir, makedirs
from os.path import join

def pickle_data(file, data_dict):
    with open(file, 'wb') as fo:
        pickle.dump(data_dict, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='bytes')
    return _dict

def main(dataset_path, path, size):
    makedirs(path, exist_ok=True)

    out_path = join(path, 'mypickle.pickle')   
    classes_names = sorted(listdir(dataset_path))

    mypickle = {"Filenames": [], "Labels": []}

    for index, classname in enumerate(classes_names):
        for image in glob.glob(join(dataset_path, classname, '*.*')):
            pil_image = Image.open(image)

            pil_image = pil_image.resize((size, size))
            pil_image.save(image)

            np_pil_image = np.asarray(Image.open(image))

            shape = np_pil_image.shape
            resolution_log2 = int(np.log2(shape[0]))

            if shape[2] not in [1, 3]:
              continue

            if shape[0] != 2**resolution_log2:
              res = 2**resolution_log2
              pil_image = pil_image.resize((res, res))
              pil_image.save(image)

            elif shape[0] != shape[1]:
              res = min(shape[0], shape[1])
              pil_image = pil_image.resize((res, res))
              pil_image.save(image)

            mypickle["Filenames"].append(image)
            mypickle["Labels"].append(index)

    pickle_data(out_path, mypickle)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-d", help="set dataset folder path")
    parser.add_argument("--size", "-s", default=512, help="set dataset folder path")
    parser.add_argument("--output_path", "-o", default='../data/', help="set output path")

    args = parser.parse_args()

    main(args.dataset_path, args.output_path, int(args.size))
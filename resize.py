import os
import shutil
import multiprocessing
from tqdm.auto import tqdm
import cv2
from PIL import UnidentifiedImageError

# ENTER YOUR INPUT DIRECTORY OF PHOTOS
INP_DIR = '/photos'
OUT_DIR = '/output'


def convert_image(name):
    """Resize image to a required shape
    :param name: name of image to load and resize
    :return: name of image in case of error
    """
    try:
        im = cv2.imread(os.path.join(INP_DIR, name))
        im = cv2.resize(im, (224, 224))
        cv2.imwrite(f'{os.getpid()}_tmp.jpg', im)
        shutil.move(f'{os.getpid()}_tmp.jpg', os.path.join(OUT_DIR, name))
    except UnidentifiedImageError:
        print('Unidentified image error. Cannot identify image file ', i)


# you can rerun script and your output will not overwrite
if __name__ == '__main__':
    files = list(set(os.listdir(INP_DIR)) - set(os.listdir(OUT_DIR)))
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for i in tqdm(pool.imap_unordered(convert_image, files)):
            pass

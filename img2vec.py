from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tqdm.auto import tqdm
import numpy as np
import pickle
import os
from PIL import UnidentifiedImageError

INP_DIR = '/output'


def get_filenames(dump_name=None):
    """Load dump data and make a list of files to transform to vectors
    :param dump_name: name of dump file
    :return: dump data and files to transform to vectors
    """
    if dump_name:
        with open(dump_name, 'rb') as f:
            result_ = pickle.load(f)
            files_ = list(set(os.listdir(INP_DIR)) - set(np.array(result_)[:, 0]))
            return result_, files_
    return list(), list(set(os.listdir(INP_DIR)))


if __name__ == '__main__':
    result, files = get_filenames()
    model = ResNet50(weights='imagenet')

    for i, img_path in enumerate(tqdm(files)):
        try:
            img = image.load_img(os.path.join(INP_DIR, img_path), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            result.append((img_path, preds,))
        except UnidentifiedImageError:
            print('Unidentified image error. Cannot identify image file ', i)
        if i % 100 == 99:
            with open('dump.pkl', 'wb') as f:
                pickle.dump(result, f)

    with open('dump.pkl', 'wb') as f:
        pickle.dump(result, f)

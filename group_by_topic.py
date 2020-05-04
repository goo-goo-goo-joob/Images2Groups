from tensorflow.keras.applications.resnet50 import decode_predictions
from tqdm.auto import tqdm
import pickle
from collections import Counter
import os
import numpy as np
import datetime
from shutil import copyfile

inp_dir = '/output'
# the number of top classes of image
TOP = 1
# the number of topics (folders) to split
NUMBER = 150

if __name__ == '__main__':
    topics = []
    all_topics = []
    classes = []

    with open('dump.pkl', 'rb') as f:
        result = pickle.load(f)

    for preds in result:
        topic = decode_predictions(preds[1], top=TOP)[0]
        topics.append((preds[0], np.array(topic)[:, 1]))

    for topic in topics:
        all_topics.extend(topic[1])

    most_topics = np.array(Counter(all_topics).most_common(NUMBER))[:, 0]

    for topic in topics:
        for k in range(1, len(topic[1])):
            if topic[1][k] in most_topics:
                classes.append((topic[0], topic[1][k]))

    gr = f'/groups_{datetime.datetime.now().strftime("%y-%m-%d_%H-%M")}'
    os.mkdir(gr)
    for i in np.unique(np.array(classes)[:, 1]):
        os.mkdir(f'{gr}/{i}')

    for nm, cl in tqdm(classes):
        copyfile(f'{inp_dir}/{nm}', f'{gr}/{cl}/{nm}')

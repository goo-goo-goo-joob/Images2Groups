import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import os
import datetime
from shutil import copyfile
from tqdm.auto import tqdm

INP_DIR = '/output'
# the number of topics (folders) to split
NUMBER = 150

if __name__ == '__main__':
    with open('dump.pkl', 'rb') as f:
        data = pickle.load(f)

    df = pd.DataFrame(data, columns=['name', 'vect'])
    df['vect'] = df['vect'].map(lambda x: x[0])
    train = np.stack(df['vect'].values)
    kmeans = MiniBatchKMeans(n_clusters=NUMBER,
                             random_state=42,
                             batch_size=100,
                             max_iter=100).fit(train)
    classes = kmeans.predict(train)
    gr = f'/groups_{datetime.datetime.now().strftime("%y-%m-%d_%H-%M")}'
    os.mkdir(gr)
    for i in np.unique(classes):
        os.mkdir(f'{gr}/{i}')

    for nm, cl in tqdm(zip(df['name'], classes)):
        copyfile(f'{INP_DIR}/{nm}', f'{gr}/{cl}/{nm}')

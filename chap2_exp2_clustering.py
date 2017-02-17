# -*- coding: utf-8 -*-  

import platform
 
def isWindowsSystem():
    return 'Windows' in platform.system()
 
def isLinuxSystem():
    return 'Linux' in platform.system()
 
if isWindowsSystem():
    pass

if isLinuxSystem():
    import redis

import numpy as np
from sklearn.cluster import KMeans
import sqlite3
import logging
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

##################
## load item ids of items with synopsis from sqlite <START>
cx = sqlite3.connect('/home/wsyj/dissertation__imdb/005_scrapy_hello/tutorial/tutorial/spiders/my_db.db')

cur = cx.cursor()
cur.execute('select distinct(item_id) from items_with_synopsis')
item_ids = cur.fetchall()
item_ids = [x[0] for x in item_ids]
print len(item_ids)

## <END>

## load word2vec model and prepare item representation <START>
model = gensim.models.Word2Vec.load('main_modelnum_features=200_min_count=5_window=2.model')
m = model

#
print('before filtering, len(item_ids): %d' % (len(item_ids)))
item_ids = filter(lambda x: str(x) in m, item_ids)
print('after filtering, len(item_ids): %d' % (len(item_ids)))

vec_list = []
for item in item_ids:
    vec_list.append(m[str(item)])

## <END>

## clustering <START>

n_digits = 77
data = np.array(vec_list)

y_pred = KMeans(init='k-means++', n_clusters=n_digits, n_init=10).fit_predict(data)
labels = y_pred
r = [(x, list(labels).count(x)) for x in set(labels)]
r.sort(key=lambda x: -1 * x[1])

print r

## <END>

def main():
    pass

if __name__ == '__main__':
    main()
    #test()

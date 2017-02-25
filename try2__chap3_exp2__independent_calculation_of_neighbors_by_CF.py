# -*- coding: utf-8 -*-  
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import gensim
from gensim import corpora, models, similarities
import os
import utility_synopsis
import math
import numpy as np
from numpy import linalg as la
from scipy.spatial.distance import cosine
import pandas as pd
import matplotlib.pyplot as plt

from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
import os
import random
import datetime
#from surprise import BaselineOnly
#from surprise import Dataset
#from surprise import evaluate
from surprise import Reader
import collections as coll

import cPickle as pickle

import heapq
import multiprocessing

count_limit = 1000	# num of users to be analyzed
N_limit = 512		# size of neighborhood

#
def extract_user_item_interaction(filename, delimiter):
    data = {}

    with open(filename , 'r') as f:
        for i, line in enumerate(f):
            userId, movieId, rating, timestamp = line.split(delimiter)
            #userId = int(userId)
            #movieId = int(movieId)
            rating = float(rating)
            timestamp = int(timestamp)

            if userId not in data:
                data[userId] = []
            data[userId].append((movieId, rating, timestamp))
    
    # order by time
    for userId in data:
        data[userId].sort(key=lambda x: x[2]) 
    return data

rating_file_name, rating_file_delimiter = os.path.sep.join(['ml-1m', 'ratings.dat']), '::'
user_item_interaction = extract_user_item_interaction(rating_file_name, rating_file_delimiter)

all_target_users = []
for i, target_user in enumerate(user_item_interaction):
    if i == count_limit:
        break
    all_target_users.append(target_user)
print 'preparation of all_target_users finished.'
#


# preparation for simple approach/CF approach
def calculate_similar_users__cf(target_user, user_history_dict):
    global N_limit

    simi_list_of_user_u = []
    interacted_items = [x[0] for x in user_history_dict[target_user]]
    for v in user_history_dict.keys():
        if target_user == v:
            #assert(False)
            continue

        user_v_history = set([x[0] for x in user_history_dict[v]])
        #print 'user_v_history:', user_v_history
        #user_u_repr = np.array(map(lambda x: 1 if x in train[u] else 0, self.distinct_item_list))
        #user_v_repr = np.array(map(lambda x: 1 if x in train[v] else 0, self.distinct_item_list))
        #common_items = user_u_history.intersection(user_v_history)
        item_union = set(interacted_items).union(user_v_history)

        if 0 == len(item_union):
            simi = 0
        else:
            #print_matrix(train[u])

            user_u_repr = np.array(map(lambda x: 1 if x in interacted_items else 0, item_union))
            user_v_repr = np.array(map(lambda x: 1 if x in user_v_history else 0, item_union))

            #print 'user_u_repr:', user_u_repr
            #print 'user_v_repr:', user_v_repr
            simi = user_u_repr.dot(user_v_repr) / (la.norm(user_u_repr * la.norm(user_v_repr)))
            #raw_input()

            #
        simi_list_of_user_u.append((v, simi))
    
    K_neighbors = heapq.nlargest(N_limit, simi_list_of_user_u, key=lambda s: s[1])    
    return K_neighbors


class InnerThreadClass(multiprocessing.Process):
    def __init__(self, name, user_item_interaction, total, partial_test_set):
        multiprocessing.Process.__init__(self)

        self.name = name
        self.user_item_interaction = user_item_interaction
        self.total = total
        self.partial_test_set = partial_test_set
        #self.N = N
 
    def run(self):
        rec = {}
        #print 'self.partial_test_set:', self.partial_test_set
        for step, user_id in enumerate(self.partial_test_set):
            #piece_of_test_data = self.partial_test_set[user_id]
            #history = piece_of_test_data[0]
            ##print 'history:', history
            #recommendation = self.recommendator.recommend(history, self.N)
            ##print 'recommendation: %s' % (str(recommendation))
            #rec[user_id] = recommendation

            ret = calculate_similar_users__cf(user_id, self.user_item_interaction)
            rec[user_id] = ret
            
            if (0 == step % 32):
                print 'progress: %d/%d' % (step, self.total)
                

        #print '[%s]: rec: %s' % (self.name, str(rec))
        pickle.dump(rec, open(self.name, 'w'))
        print 'done'


def calculate_neighbors__cf(all_target_users):
    threads = []
    # Start consumers
    num_threads = multiprocessing.cpu_count() * 2
    print 'Creating %d threads' % num_threads
    piece_len = len(all_target_users) / num_threads

    pieces = []
    user_id_list = list(all_target_users)
    pieces = [user_id_list[x * piece_len: (x + 1) * piece_len] for x in xrange(0, num_threads + 1)]


    # 创建线程对象
    name_prefix = 'chap3_exp2_thread-'
    name_postfix = '.dump'
    for en, x in enumerate(pieces):
        name = name_prefix + str(en) + name_postfix
        threads.append(InnerThreadClass(name, user_item_interaction, len(user_id_list), x))
    for t in threads:
        t.start()
    for t in threads:
        t.join()  

    total = len(user_id_list)
    print 'progress: %d/%d. done.' % (total, total)
    
    W_pieces = [pickle.load(open(name, 'r')) for name in [name_prefix + str(en) + name_postfix for en, x in enumerate(pieces)]]
    W = {}
    map(lambda x: W.update(x), W_pieces)

    print 'going to dump final dict'
    starttime = datetime.datetime.now()
    pickle.dump(W, open(name_prefix + 'final_dump' + name_postfix, 'w'))
    endtime = datetime.datetime.now()
    interval = (endtime - starttime).seconds
    print 'metric calculation: time consumption: %d' % (interval)
    return W



starttime = datetime.datetime.now()

calculate_neighbors__cf(all_target_users)

endtime = datetime.datetime.now()
interval = (endtime - starttime).seconds
print 'metric calculation: time consumption: %d' % (interval)

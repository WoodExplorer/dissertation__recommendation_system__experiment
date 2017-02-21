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

import os
import gensim#from gensim.models import word2vec
import math
import random
import csv
import numpy as np
from numpy import linalg as la
import heapq
import multiprocessing
#from multiprocessing.dummy import Pool as ThreadPool
import datetime
import logging
#import threading
#import thread
import cPickle
import sqlite3

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_least_numbers_big_data(self, alist, k):
    max_heap = []
    length = len(alist)
    if not alist or k <= 0 or k > length:
        return
    k = k - 1
    for ele in alist:
        ele = -ele
        if len(max_heap) <= k:
            heapq.heappush(max_heap, ele)
        else:
            heapq.heappushpop(max_heap, ele)

    return map(lambda x:-x, max_heap)

def extract_data_from_file_and_generate_train_and_test(filename, train_percent, seed, delimiter, test_data_inner_ratio, sort_by_time=False):
    test = None
    train = None
    data = {}
    random.seed(seed)

    with open(filename , 'r') as f:
        first_line = f.readline()
        for i, line in enumerate(f):
            userId, movieId, rating, timestamp = line.split(delimiter)
            #userId = int(userId)
            #movieId = int(movieId)
            rating = float(rating)
            timestamp = int(timestamp)

            if userId not in data:
                data[userId] = []
            data[userId].append((movieId, rating, timestamp))

    test = {}
    train = {}
    for userId in data:
        total_len = len(data[userId])
        if random.random() >= train_percent:
        #if 0 == random.randint(0, 2):
        #if 2 == random.randint(0, 2):
            test[userId] = data[userId]
        else:
            train[userId] = data[userId]
    userId = None

    for userId in test:
        test[userId].sort(key=lambda x: x[2])
    # sort by time: PART 1<begin>
    if sort_by_time:
        print 'sort by time PART 1'
        for userId in train:
            train[userId].sort(key=lambda x: x[2])

    #print train[train.keys()[0]]
    #print test[test.keys()[0]]
    #raw_input()
    # sort by time: PART 1 <end>

    ### split test data further
    test_real = {}
#    if not sort_by_time:
#        for k_user in test:
#            test_real[k_user] = [[], []]
#            for m, r, t in test[k_user]:
#                # every record of test dataset is supposed to be splitted into 2 parts: input part and fact/answer part
#                # How to specify the relative of these parts? 
#                #  Assign num_of_parts_of_test_data an appropriate value: each record of test dataset is supposed
#                # to constitute num_of_parts_of_test_data parts, and one of them would serve as the fact/answer part.
#                if 0 == random.randint(0, num_of_parts_of_test_data):
#                    test_real[k_user][1].append((m, r, t)) # the fact/answer part in one record of test dataset
#                else:
#                    test_real[k_user][0].append((m, r, t)) # the input part in one record of test dataset
#    else: 
#        # sort by time: PART 2 <start>
#        print 'sort by time PART 2'
    for k_user in test:
        #print 'len(test[k_user]):', len(test[k_user])
        #print 'num_of_parts_of_test_data:', num_of_parts_of_test_data
        #raw_input()
        #print (len(test[k_user]) * (1.0 / num_of_parts_of_test_data))
        #print (int)(len(test[k_user]) * (1.0 / num_of_parts_of_test_data))
        split_point_index = -1 * ((int)(len(test[k_user]) * test_data_inner_ratio))
        #print 'split_point_index:', split_point_index
        test_real[k_user] = [test[k_user][:split_point_index], test[k_user][split_point_index:]]
        #raw_input()

        # sort by time: PART 2 <end>

    #print test_real[test_real.keys()[0]]
    print 'sort_by_time:', sort_by_time

    #raw_input('pause')

    return train, test_real

class RecommendatorSystem(object):
    """docstring for RecommendatorSystem"""
    def __init__(self):
        super(RecommendatorSystem, self).__init__()
        
    def setup(self):
        assert(False)

    def split_data(self, data, M, k, seed):
        '''Possible problem: 
        '''
        test = []
        train = []
        random.seed(seed)
        for user, item in data.items():
            if k == random.randint(0, M):
                test.append([user, item])
            else:
                train.append([user, item])
        return train, test


    def calculate_metrics(self, train, test, N):
        starttime = datetime.datetime.now()

        ###
        threads = []
        # Start consumers
        num_threads = multiprocessing.cpu_count() * 2
        #num_threads = 1 # for debugging        
        if len(test) < num_threads:
            num_threads = 1
        
        print 'Creating %d threads' % num_threads

        piece_len = len(test) / num_threads

        test__in_list = test.items()
        #print 'dict(test__in_list):', dict(test__in_list)
        pieces = [dict(test__in_list[x * piece_len: (x + 1) * piece_len]) for x in xrange(0, num_threads + 1)]
        #print 'pieces:', pieces
        # 创建线程对象
        name_prefix = 'thread-'
        name_postfix = '.dump'
        for en, x in enumerate(pieces):
            name = name_prefix + str(en) + name_postfix
            threads.append(InnerThreadClass(name, self, len(test), x, N))
        for t in threads:
            t.start()
        for t in threads:
            t.join()  

        total = len(test)
        print 'progress: %d/%d. done.' % (total, total)

        #
        rec_pieces = [cPickle.load(open(name, 'r')) for name in [name_prefix + str(en) + name_postfix for en in xrange(num_threads + 1)]]
        #print 'rec_pieces:', rec_pieces
        rec = {}
        map(lambda x: rec.update(x), rec_pieces)
        
        #print 'rec:', rec
        ###

        hit = 0
        all__for_recall = 0
        all__for_precision = 0
        for user in test.keys():
            history = test[user][0]
            answer = test[user][1]
            tu = [x[0] for x in answer]
            rank = rec[user] # self.recommend(history, N)
            #print 'rank:', rank
            for item, pui in rank:
                if item in tu:
                    hit += 1
            all__for_recall += len(tu)
            all__for_precision += len(rank) #Note: In book RSP, the author used 'all += N'
        
        metric_recall = None
        metric_precision = None
        metric_f1 = None
        if 0 == all__for_recall:
            metric_recall = 0
        else:
            metric_recall = hit / (all__for_recall * 1.0)
        
        if 0 == all__for_precision:
            metric_precision = 0
        else:
            metric_precision = hit / (all__for_precision * 1.0)

        if 0 == all__for_recall or 0 == all__for_precision:
            metric_f1 = 0
        else:
            metric_f1 = 2/(1./metric_precision + 1./metric_recall)

        endtime = datetime.datetime.now()
        interval = (endtime - starttime).seconds
        print 'metric calculation: time consumption: %d' % (interval)
        return {'recall': metric_recall, 'precision': metric_precision, 'f1': metric_f1}

class InnerThreadClass(multiprocessing.Process):
    def __init__(self, name, recommendator, total, partial_test_set, N):
        multiprocessing.Process.__init__(self)

        self.name = name
        self.recommendator = recommendator
        self.total = total
        self.partial_test_set = partial_test_set
        self.N = N
 
    def run(self):
        rec = {}
        #print 'self.partial_test_set:', self.partial_test_set
        for step, user_id in enumerate(self.partial_test_set):
            piece_of_test_data = self.partial_test_set[user_id]
            history = piece_of_test_data[0]
            #print 'history:', history
            recommendation = self.recommendator.recommend(history, self.N)
            #print 'recommendation: %s' % (str(recommendation))
            rec[user_id] = recommendation

            if (0 == step % 64):
                print 'progress: %d/%d' % (step, self.total)

        #print '[%s]: rec: %s' % (self.name, str(rec))
        cPickle.dump(rec, open(self.name, 'w'))
        print 'done'


class RecommendatorSystemViaCollaborativeFiltering(RecommendatorSystem):
    """docstring for RecommendatorSystemViaCollaborativeFiltering"""
    def __init__(self):
        super(RecommendatorSystemViaCollaborativeFiltering, self).__init__()
        self.W = None   # weight matrix / user similarity matrix

    def setup(self, para):
        self.train = para['train']

        # K
        self.K = para['K']

        #self.user_similarity(self.train)


    def user_similarity(self, train):
        #build inverse table item_users
        print 'This is RecommendatorSystemViaCollaborativeFiltering'
        starttime = datetime.datetime.now()
        
        ###
        #
        #user set
        user_id_set = train.keys()

        #user history dict
        #user_history = {x: data[x].keys() for x in data}
        #print 'user_history:', user_history
        user_history = train

        #
        #calculate final similarity matrix W
        threads = []
        # Start consumers
        num_threads = multiprocessing.cpu_count() * 2
        print 'Creating %d threads' % num_threads
        piece_len = len(user_id_set) / num_threads

        pieces = []
        user_id_list = list(user_id_set)
        pieces = [user_id_list[x * piece_len: (x + 1) * piece_len] for x in xrange(0, num_threads + 1)]


        # 创建线程对象
        name_prefix = 'thread-'
        name_postfix = '.dump'
        for en, x in enumerate(pieces):
            name = name_prefix + str(en) + name_postfix
            threads.append(InnerThreadClass(name, train, x, self.K))
        for t in threads:
            t.start()
        for t in threads:
            t.join()  

        total = len(user_id_list)
        print 'progress: %d/%d. done.' % (total, total)

        #
        W_pieces = [cPickle.load(open(name, 'r')) for name in [name_prefix + str(en) + name_postfix for en, x in enumerate(pieces)]]
        W = {}
        map(lambda x: W.update(x), W_pieces)
        self.W = W

        #
        endtime = datetime.datetime.now()
        interval=(endtime - starttime).seconds
        print 'user_similarity: time consumption: %d' % (interval)
        

    def find_K_neighbors(self, target_user_history, K):
        ### find K neighbors <begin>
        simi_list_of_user_u = []
        interacted_items = [x[0] for x in target_user_history]
        for v in self.train.keys():
            #if u == v:
            #    assert(False)
            #    continue

            user_v_history = set([x[0] for x in self.train[v]])
            #print 'user_v_history:', user_v_history
            #user_u_repr = np.array(map(lambda x: 1 if x in train[u] else 0, self.distinct_item_list))
            #user_v_repr = np.array(map(lambda x: 1 if x in train[v] else 0, self.distinct_item_list))
            #common_items = user_u_history.intersection(user_v_history)
            common_items = set(interacted_items).union(user_v_history)

            if 0 == len(common_items):
                simi = 0
            else:
                #print_matrix(train[u])

                user_u_repr = np.array(map(lambda x: 1 if x in interacted_items else 0, common_items))
                user_v_repr = np.array(map(lambda x: 1 if x in user_v_history else 0, common_items))

                #print 'user_u_repr:', user_u_repr
                #print 'user_v_repr:', user_v_repr
                simi = user_u_repr.dot(user_v_repr) / (la.norm(user_u_repr * la.norm(user_v_repr)))
                #raw_input()

                #
            simi_list_of_user_u.append((v, simi))

            #
        K_neighbors = heapq.nlargest(self.K * 2, simi_list_of_user_u, key=lambda s: s[1])
        ### find K neighbors <end>
        return K_neighbors

    def recommend(self, target_user_history, N, K=10):
        '''@N: number of user neighbors considered
        '''
        rank = {}
        interacted_items = [x[0] for x in target_user_history]
        #print 'target_user_history:', target_user_history
        #print 'interacted_items:', interacted_items

        K_neighbors = self.find_K_neighbors(target_user_history, K)

        for v, wuv in K_neighbors:
        #for v, wuv in sorted(self.W[u].items(), key=lambda x: x[1], reverse=True)[0:K]: # wuv: similarity between user u and user v
            for i, rvi, timestamp in self.train[v]: # rvi: rate of item by user v
                if i in interacted_items:
                    #do not recommend items which user u interacted before
                    continue

                if i not in rank:
                    rank[i] = 0.0
                rank[i] += wuv * rvi

        rank = rank.items()
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank[:N]

### Word2vec
class RecommendatorViaWord2Vec(RecommendatorSystemViaCollaborativeFiltering):
    """docstring for RecommendatorViaWord2Vec"""
    def __init__(self):
        super(RecommendatorViaWord2Vec, self).__init__()
        #self.model = None
        self.W = None

    def setup(self, para):
        data = para['data']
        self.train = para['data']
        model_name = para['model_name'] if 'model_name' in para else 'tmp_model'
        num_features = para['num_features']
        min_count = para['min_count']
        window = para['window']
        para_iter = para['iter']
        batch_words = para['batch_words']

        self.K = para['K']

        model_name += ('_'.join(['num_features=' + str(num_features), 'min_count=' + str(min_count), 'window=' + str(window), 'iter=' + str(para_iter)]) + '.model')

        #list_of_list = convert_2_level_dict_to_list_of_list(data)
        list_of_list = convert_level_1_dict_level_2_list_of_size_3_tuples_to_list_of_list(data)
        #print 'list_of_list:', list_of_list

        print 'start training'
        self.model = gensim.models.Word2Vec(list_of_list, size=num_features, min_count=min_count, window=window, sg=0, iter=para_iter, batch_words=batch_words)
        print 'training finished'

        # If you don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        self.model.init_sims(replace=True)

        # It can be helpful to create a meaningful model name and 
        # save the model for later use. You can load it later using Word2Vec.load()
        self.model.save(model_name)

        #
        #user set
        user_id_set = data.keys()

        #user history dict
        #user_history = {x: data[x].keys() for x in data}
        #print 'user_history:', user_history

        #user repre dict
        #user_repre = {uesr_id: np.average(map(lambda item: self.model[item], user_history[uesr_id]), axis=0) for uesr_id in user_history}
        self.user_repre = {uesr_id: user_history2user_repr(self.model, data[uesr_id]) for uesr_id in data}
        #print 'user_repre:', user_repre

        
    def find_K_neighbors(self, target_user_history, K):
        ### find K neighbors <begin>
        simi_list_of_user_u = []
        #print 'interacted_items:', interacted_items
        user_repre_of_u = user_history2user_repr(self.model, target_user_history)

        for v in self.train.keys():
            #if u == v:
            #    assert(False)
            #    continue

            user_v_history = self.user_repre[v]
            simi = user_repre_of_u.dot(self.user_repre[v]) / (la.norm(user_repre_of_u * la.norm(self.user_repre[v])))

                #
            simi_list_of_user_u.append((v, simi))

            #
        K_neighbors = heapq.nlargest(self.K * 2, simi_list_of_user_u, key=lambda s: s[1])
        ### find K neighbors <end>
        return K_neighbors

def user_history2user_repr(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = map(lambda x: (model[x[0]], x[1], x[2]), items_existed_in_model)
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    items_multiplied_by_rate = map(lambda (vec, rate, timestamp): vec * rate, items_translated_to_vecs)
    #print 'items_multiplied_by_rate:', items_multiplied_by_rate[0]
    #raw_input()
    return np.average(items_multiplied_by_rate, axis=0)
  

def print_matrix(M):
    def print_wrapper(x):
        print x, M[x]
    map(lambda x: print_wrapper(x), M)


#def extract_data():
#    filename = 'ml-latest-small\\ratings.csv'
#    data = {}
#
#    with open(filename , 'r') as f:
#        first_line = f.readline()
#        for i, line in enumerate(f):
#            userId, movieId, rating, timestamp = line.split(',')
#            userId = int(userId)
#            movieId = int(movieId)
#            rating = float(rating)
#
#            if userId not in data:
#                data[userId] = {}
#            data[userId][movieId] = rating
#
#            if 10 == i:
#                break
#
#    #print data
#    return data
#
#
#    #csvfile = file(filename, 'r')
#    #reader = csv.reader(csvfile)
#    #
#    #for line in reader:
#    #    print line
#    #
#    #csvfile.close() 


def main_windows():
    mode_dict = {
        1: 'RecommendatorSystemViaCollaborativeFiltering',
        2: 'RecommendatorViaWord2Vec',
        3: 'RecommendatorViaDoc2Vec',
        4: 'RecommendatorViaDoc2Vec time'
    }
    print mode_dict
    mode = int(raw_input('Please select mode:'))
    if 1 == mode:
        #data_filename, delimiter = os.path.sep.join(['ml-latest-small', 'ratings.csv']), ','
        data_filename, delimiter = os.path.sep.join(['ml-100k', 'u.data']), '\t'

        seed = 2 
        train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 4, 0, seed, delimiter)


        rs = RecommendatorSystemViaCollaborativeFiltering()
        K = 10
        
        for N in xrange(10, 11):
        #for N in xrange(3, 50):
            print 'N:', N


            rs.setup({'train': train, 'K': K})

            metrics = rs.calculate_metrics(train, test, N)
            print 'metrics:', metrics
    elif 2 == mode:
        #data_filename, delimiter = os.path.sep.join(['ml-latest-small', 'ratings.csv']), ','
        #data_filename, delimiter = os.path.sep.join(['ml-1m', 'ratings.dat']), '::'
        #data_filename, delimiter = os.path.sep.join(['ml-10M100K', 'ratings.dat']), '::'
        data_filename, delimiter = os.path.sep.join(['ml-100k', 'u.data']), '\t'

        K = 10
        seed = 2 
        train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 4, 0, seed, delimiter)

        rs = RecommendatorViaWord2Vec()
        rs.setup({'data': train, 
            'model_name': data_filename + '_' + 'main_word2vec_model',
            'num_features': 100,
            'min_count': 1,
            'window': 5,
            'K': K,
        })

        N = 10
        metrics = rs.calculate_metrics(train, test, N)
        print 'metrics:', metrics
    elif 3 == mode:
        #data_filename, delimiter = os.path.sep.join(['ml-latest-small', 'ratings.csv']), ','
        #data_filename, delimiter = os.path.sep.join(['ml-1m', 'ratings.dat']), '::'
        #data_filename, delimiter = os.path.sep.join(['ml-10M100K', 'ratings.dat']), '::'
        data_filename, delimiter = os.path.sep.join(['ml-100k', 'u.data']), '\t'

        K = 10
        seed = 2 
        dm = 0
        train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 4, 0, seed, delimiter)

        rs = RecommendatorViaDoc2Vec()
        rs.setup({'data': train, 
            'model_name': data_filename + '_' + 'main_doc2vec_model',
            'num_features': 100,
            'min_count': 3,
            'window': 2,
            'K': K,
            'dm': dm,
        })

        N = 10
        
        metrics = rs.calculate_metrics(train, test, N)
        print 'metrics:', metrics
    elif 4 == mode:
        #data_filename, delimiter = os.path.sep.join(['ml-latest-small', 'ratings.csv']), ','
        #data_filename, delimiter = os.path.sep.join(['ml-1m', 'ratings.dat']), '::'
        #data_filename, delimiter = os.path.sep.join(['ml-10M100K', 'ratings.dat']), '::'
        data_filename, delimiter = os.path.sep.join(['ml-100k', 'u.data']), '\t'

        K = 10
        seed = 2
        dm = 0
        #test_set_ratio = 0.2
        train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 4, 0, seed, delimiter, sort_by_time=True)

        rs = RecommendatorViaDoc2Vec()
        rs.setup({'data': train, 
            'model_name': data_filename + '_' + 'main_doc2vec_model',
            'num_features': 100,
            'min_count': 3,
            'window': 20,
            'K': K,
            'dm': dm,
        })

        N = 10
        
        metrics = rs.calculate_metrics(train, test, N)
        print 'metrics:', metrics


def try_different_train_test_ratio(ttratio, test_data_inner_ratio): # ttratio: train test ratio
    cx = sqlite3.connect('my_metrics.db')
    cur = cx.cursor()

    data_filename, delimiter = os.path.sep.join(['ml-100k', 'u.data']), '\t'

    seed = 2 
    K = 10
    N = 20

    train, test = extract_data_from_file_and_generate_train_and_test(data_filename, ttratio, seed, delimiter, test_data_inner_ratio)
    #train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 3, 0, seed, delimiter)

    para_iter = 35
    batch_words = 10000
    data_set = '1M'
    table_name_prefix = 'ttratio_tiratio__metrics_N_%d__iter_%d__batch_words_%d__da_%s'

    table_name = table_name_prefix % (N, para_iter, batch_words, data_set)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='%s';" % table_name)
    ret = cur.fetchall()
    if 0 == len(ret):
        sql = '''create table %s (
  _row_ID integer	primary key autoincrement,
  
  ttratio decimal(10, 9),
  test_data_inner_ratio decimal(10, 9),

  size integer,
  min_count integer,
  window integer,

  precision decimal(30, 28),
  recall decimal(30, 28),
  f1 decimal(30, 28),
  
  CreatedTime TimeStamp NOT NULL DEFAULT (datetime('now','localtime'))
);''' % (table_name)
        cur.execute(sql)
        cx.commit()

    para_combs = [[440, 1, 2]]
    print para_combs[0]
    
    for i, (s, mc, w) in enumerate(para_combs):
        rs = RecommendatorViaWord2Vec()
        rs.setup({'data': train, 
            'model_name': 'main_model',
            'num_features': s,
            'min_count': mc,
            'window': w,
            'K': K,
            'iter': para_iter,
            'batch_words': batch_words,

        })

        
        metrics = rs.calculate_metrics(train, test, N)
        print metrics
        precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1']
        
        cur.execute('insert into %s (ttratio, test_data_inner_ratio, size, min_count, window, precision, recall, f1)' % (table_name) +
                   'values (%.19f, %.19f, %d, %d, %d, %.19f, %.19f, %.19f)' % (ttratio, test_data_inner_ratio, s, mc, w, precision, recall, f1))

        cx.commit()

    ## CF <START> #########################################################
    rs = RecommendatorSystemViaCollaborativeFiltering()
    #rs = RecommendatorSystemViaCollaborativeFiltering_UsingRedis()

    rs.setup({
        'train': train,
        'K': K,
    })

    metrics = rs.calculate_metrics(train, test, N)
    print 'metrics:', metrics

    precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1']
        
    cur.execute('insert into %s (ttratio, test_data_inner_ratio, size, min_count, window, precision, recall, f1)' % (table_name) +
                   'values ( %.19f, %.19f, %d, %d, %d, %.19f, %.19f, %.19f)' % (ttratio, test_data_inner_ratio, -1, -1, -1, precision, recall, f1))
    cx.commit()
    ## CF <END> #########################################################
    #exit(0)
    ###

    cur.close()
    cx.close()
    

def wrapper__try_different_ttratio_and_tiratio():
    #train_test_ratio_list = [2. / 3]
    #train_test_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    #train_test_ratio_list = [0.05, 0.06, 0.07, 0.08, 0.09]
    #train_test_ratio_list = [0.01, 0.02, 0.03, 0.04, 0.05]
    #for train_test_ratio in train_test_ratio_list:
    #    try_different_train_test_ratio(train_test_ratio)

    #for train_test_ratio, test_data_inner_ratio in [(0.5, 0.80), (0.5, 0.85), (0.5, 0.9), (0.5, 0.95)]:
    #for train_test_ratio, test_data_inner_ratio in [(0.5, 0.05), (0.5, 0.06), (0.5, 0.07), (0.5, 0.08)]:
    for train_test_ratio, test_data_inner_ratio in [(0.02, 0.05), (0.03, 0.05), (0.04, 0.05), (0.05, 0.05)]:
        try_different_train_test_ratio(train_test_ratio, test_data_inner_ratio)

def main_Linux():
    #data_filename, delimiter = os.path.sep.join(['ml-latest-small', 'ratings.csv']), ','
    #data_filename, delimiter, data_set = os.path.sep.join(['ml-1m', 'ratings.dat']), '::', '1M'
    #data_filename, delimiter = os.path.sep.join(['ml-10M100K', 'ratings.dat']), '::'
    data_filename, delimiter, data_set = os.path.sep.join(['ml-100k', 'u.data']), '\t', '100K'

    seed = 2 
    K = 10
    test_data_inner_ratio = 0.25
    train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 2. / 3, seed, delimiter, test_data_inner_ratio)
    #train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 3, 0, seed, delimiter)

    ## CF <START>
    '''rs = RecommendatorSystemViaCollaborativeFiltering()
    #rs = RecommendatorSystemViaCollaborativeFiltering_UsingRedis()

    rs.setup({
        'train': train,
        'K': K,
    })
    
    for N in xrange(20, 21):
    #for N in xrange(10, 11):
    #for N in xrange(3, 50):
        print 'N:', N

        metrics = rs.calculate_metrics(train, test, N)
        print 'metrics:', metrics
    ## CF <END>
    exit(0)'''
    ###
    N = 20
    para_iter = 35
    batch_words = 10000
    table_name_prefix = 'metrics_N_%d__iter_%d__batch_words_%d__da_%s'

    cx = sqlite3.connect('my_metrics.db')
    cur = cx.cursor()

    table_name = table_name_prefix % (N, para_iter, batch_words, data_set)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='%s';" % table_name)
    ret = cur.fetchall()
    if 0 == len(ret):
        sql = '''create table %s (
  _row_ID integer	primary key autoincrement,
  
  size integer,
  min_count integer,
  window integer,

  precision decimal(30, 28),
  recall decimal(30, 28),
  f1 decimal(30, 28),
  
  CreatedTime TimeStamp NOT NULL DEFAULT (datetime('now','localtime'))
);''' % (table_name)
        cur.execute(sql)
        cx.commit()

    para_size = range(100, 501, 10)
    para_min_count = range(1, 2, 1)
    para_window = range(3, 4, 1)

    #para_combs = zip(para_size, para_min_count, para_window)
    #para_combs = [[[(s, mc, w) for w in para_window] for mc in para_min_count] for s in para_size]
    para_combs = [(s, mc, w) for w in para_window for mc in para_min_count for s in para_size]
    #para_combs = [[480, 1, 4]]
    print para_combs[0]
    
    for i, (s, mc, w) in enumerate(para_combs):
        print "loop %d/%d" % (i, len(para_combs))
        #if (i < 215):
        #    continue
        

        rs = RecommendatorViaWord2Vec()
        rs.setup({'data': train, 
            'model_name': 'main_model',
            'num_features': s,
            'min_count': mc,
            'window': w,
            'K': K,
            'iter': para_iter,
            'batch_words': batch_words,

        })

        
        metrics = rs.calculate_metrics(train, test, N)
        print metrics
        precision, recall, f1 = metrics['precision'], metrics['recall'], metrics['f1']
        
        cur.execute('insert into %s (size, min_count, window, precision, recall, f1)' % (table_name) +
                   'values (%d, %d, %d, %.19f, %.19f, %.19f)' % (s, mc, w, precision, recall, f1))

        cx.commit()
    cur.close()
    cx.close()
    



    ###
#    N = 10
#    K = 10
#    dm = 0
#
#    rs = RecommendatorViaDoc2Vec()
#    rs.setup({'data': train, 
#        'model_name': data_filename + '_' + 'main_doc2vec_model',
#        'num_features': 300,
#        'min_count': 3,
#        'window': 20,
#        'K': K,
#        'dm': dm,
#    })
#
#    
#    metrics = rs.calculate_metrics(train, test, N)
#    print 'metrics:', metrics

def main():
    if isWindowsSystem():
        main_windows()
        return

    if isLinuxSystem():
        wrapper__try_different_ttratio_and_tiratio()
#        main_Linux()
        return


def function(sentences):
    assert(False)
    # Set values for various parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 40   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    
    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context.model"
    model.save(model_name)


def convert_2_level_dict_to_list_of_list(data):
    return [data[x].keys() for x in data]


def convert_2_level_dict_to_list_of_LabeledSentence(data):
    return [gensim.models.doc2vec.LabeledSentence(data[x].keys(), [x]) for x in data]

def convert_level_1_dict_level_2_list_of_size_3_tuples_to_list_of_LabeledSentence(data):
    return [gensim.models.doc2vec.LabeledSentence(map(lambda y: y[0], data[x]), [x]) for x in data]

def convert_level_1_dict_level_2_list_of_size_3_tuples_to_list_of_list(data):
    return [map(lambda y: y[0], data[x]) for x in data]

def test():
#    train = {'A': {'a': 1, 'b': 1, 'd': 1},
#        'B': {'a': 1, 'c': 1},
#        'C': {'b': 1, 'e': 1},
#        'D': {'c': 1, 'd': 1, 'e': 1},}
#
#    #test = {'A': {'a': 1, 'b': 1, 'd': 1}, }
#    test = {'A': {'c': 1}, }
#
#    #list_of_list = convert_2_level_dict_to_list_of_list(train)
#    #print 'list_of_list:', list_of_list
#
#    #rs = RecommendatorViaWord2Vec()
#    #rs.setup({'data': train, 'model_name': 'test_model'})
#
#


    train = {
    #train = {'A': [('a', 1, 1), ('b', 1, 2), ('d', 1, 3)],
        'B': [('a', 1, 1), ('c', 1, 2)],
        'C': [('b', 1, 1), ('e', 1, 2)],
        'D': [('c', 1, 1), ('d', 1, 2), ('e', 1, 3)],}

    #test = {'A': {'a': 1, 'b': 1, 'd': 1}, }
    test = {'A': [[('a', 1, 1), ('b', 1, 2), ('d', 1, 3)], [('c', 1, 4)]], }

    ###
    mode_dict = {
        1: 'RecommendatorSystemViaCollaborativeFiltering',
        2: 'RecommendatorViaWord2Vec',
        3: 'RecommendatorViaDoc2Vec',
        4: 'RecommendatorViaDoc2Vec time'
    }
    print mode_dict
    mode = int(raw_input('Please select mode:'))
    if 1 == mode:
        rs = RecommendatorSystemViaCollaborativeFiltering()
        rs.setup({'train': train, 'K': 10})
    
        #print 'rs.W:'
        #print rs.W
        #print_matrix(rs.W)
    
        N = 10
        
        metrics = rs.calculate_metrics(train, test, N)
        print 'metrics:', metrics
    elif 4 == mode:

        #list_of_list = convert_2_level_dict_to_list_of_list(train)
        #print 'list_of_list:', list_of_list

        #rs = RecommendatorViaWord2Vec()
        #rs.setup({'data': train, 'model_name': 'test_model'})

        N = 10
        K = 10

        rs = RecommendatorViaDoc2Vec()
        rs.setup({'data': train, 
            'num_features': 3,
            'min_count': 0,
            'window': 20,
            'K': K, })

        
        metrics = rs.calculate_metrics(train, test, N)
        print 'metrics:', metrics


if __name__ == '__main__':
    main()
    #test()

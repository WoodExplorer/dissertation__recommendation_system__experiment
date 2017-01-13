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

def extract_data_from_file_and_generate_train_and_test(filename, M, k, seed, delimiter):
    test = {}
    train = {}
    random.seed(seed)

    with open(filename , 'r') as f:
        first_line = f.readline()
        for i, line in enumerate(f):
            userId, movieId, rating, timestamp = line.split(delimiter)
            #userId = int(userId)
            #movieId = int(movieId)
            rating = float(rating)

            if k == random.randint(0, M):
                if userId not in test:
                    test[userId] = {}
                test[userId][movieId] = rating
            else:
                if userId not in train:
                    train[userId] = {}
                train[userId][movieId] = rating

    return train, test


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


    def recall(self, train, test, N):
        starttime = datetime.datetime.now()

        hit = 0
        all = 0
        for user in test.keys():
            tu = test[user]
            rank = self.recommend(user, train, N)
            #print 'rank:', rank
            for item, pui in rank:
                if item in tu:
                    hit += 1
            all += len(tu)
        metric = hit / (all * 1.0)

        endtime = datetime.datetime.now()
        interval = (endtime - starttime).seconds
        print 'recall: time consumption: %d' % (interval)
        return metric

    def precision(self, train, test, N):
        starttime = datetime.datetime.now()

        hit = 0
        all = 0
        for user in test.keys():
            tu = test[user]
            rank = self.recommend(user, train, N)
            for item, pui in rank:
                if item in tu:
                    hit += 1
            all += len(rank) #Note: In book RSP, the author used 'all += N'
        metric = hit / (all * 1.0)

        endtime = datetime.datetime.now()
        interval=(endtime - starttime).seconds
        print 'precision: time consumption: %d' % (interval)
        return metric

class InnerThreadClass(multiprocessing.Process):
    def __init__(self, name, train, target_user_id_list, K):
        multiprocessing.Process.__init__(self)

        self.target_user_id_list = target_user_id_list
        self.whole_user_id_list = train.keys()
        self.train = train
        self.W = {}
        self.K = K
        self.name = name

        self.total = len(target_user_id_list)
 
    def run(self):
        map(lambda (step, u): self.inner(step, u), enumerate(self.target_user_id_list)) 
        cPickle.dump(self.W, open(self.name, 'w'))
        print 'done'

    def inner(self, step, u):
        user_u_history = set(self.train[u].keys())
        simi_list_of_user_u = []
        for v in self.whole_user_id_list:
            if u == v:
                continue

            user_v_history = set(self.train[v].keys())
            #user_u_repr = np.array(map(lambda x: 1 if x in train[u] else 0, self.distinct_item_list))
            #user_v_repr = np.array(map(lambda x: 1 if x in train[v] else 0, self.distinct_item_list))
            #common_items = user_u_history.intersection(user_v_history)
            common_items = user_u_history.union(user_v_history)

            if 0 == len(common_items):
                simi = 0
            else:
                #print_matrix(train[u])

                user_u_repr = np.array(map(lambda x: 1 if x in self.train[u] else 0, common_items))
                user_v_repr = np.array(map(lambda x: 1 if x in self.train[v] else 0, common_items))

                #print 'user_u_repr:', user_u_repr
                #print 'user_v_repr:', user_v_repr
                simi = user_u_repr.dot(user_v_repr) / (la.norm(user_u_repr * la.norm(user_v_repr)))
                #raw_input()

                #
            simi_list_of_user_u.append((v, simi))

            #
        K_neighbors = heapq.nlargest(self.K * 2, simi_list_of_user_u, key=lambda s: s[1])
        #K_neighbors = sorted(simi_list_of_user_u, key=lambda x: x[1], reverse=True)[0:self.K]
        #print 'K_neighbors:', K_neighbors
        #raw_input()



        self.W[u] = dict(K_neighbors)
        


        if (0 == step % 64):
            print 'progress: %d/%d' % (step, self.total)


class RecommendatorSystemViaCollaborativeFiltering(RecommendatorSystem):
    """docstring for RecommendatorSystemViaCollaborativeFiltering"""
    def __init__(self):
        super(RecommendatorSystemViaCollaborativeFiltering, self).__init__()
        self.W = None   # weight matrix / user similarity matrix

    def setup(self, para):
        train = para['train']

        # K
        self.K = para['K']

        self.user_similarity(train)




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

        W_sub = {}


        

        #map(lambda (step, u): inner(step, u), enumerate(user_id_set)) 

        ## Make the Pool of workers
        #pool = ThreadPool(4) 
        ## Open the urls in their own threads
        ## and return the results
        #results = pool.map(lambda (step, u): inner(step, u), enumerate(user_id_set))
        ##close the pool and wait for the work to finish 
        #pool.close() 
        #pool.join()


        threads = []
        num = 4
        piece_len = len(user_id_set) / num

        pieces = []
        user_id_list = list(user_id_set)
        pieces = [user_id_list[x * piece_len: (x + 1) * piece_len] for x in xrange(0, num + 1)]


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
        

    def recommend(self, u, train, N, K=10):
        '''@N: number of user neighbors considered
        '''
        rank = {}
        interacted_items = train[u]
        for v, wuv in sorted(self.W[u].items(), key=lambda x: x[1], reverse=True)[0:K]: # wuv: similarity between user u and user v
            for i, rvi in train[v].items(): # rvi: rate of item by user v
                if i in interacted_items:
                    #do not recommend items which user u interacted before
                    continue

                if i not in rank:
                    rank[i] = 0.0
                rank[i] += wuv * rvi

        rank = rank.items()
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank[:N]


class RecommendatorSystemViaCollaborativeFiltering_UsingRedis(RecommendatorSystemViaCollaborativeFiltering):
    """docstring for RecommendatorSystemViaCollaborativeFiltering_UsingRedis"""
    def __init__(self):
        super(RecommendatorSystemViaCollaborativeFiltering_UsingRedis, self).__init__()
        
        #
        self.my_redis = redis.Redis(host='localhost', port=6379, db=0) 

    def setup(self, para):
        self.user_similarity(para['train'])

    def user_similarity(self, train):
        #build inverse table item_users
        item_users = {}
        for u, items in train.items():
            for i in items.keys():
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)

        #calculate co-rated items between users
        C = {}
        #C_prefix = 'C_'

        N = {}
        for i, users in item_users.items():
            for u in users:
                if u not in N:
                    N[u] = 0
                N[u] += 1

                for v in users:
                    if u == v:
                        continue

                    if u not in C:
                        C[u] = {}
                    if v not in C[u]:
                        C[u][v] = 0
                    C[u][v] += 1
        print 'C matrix calculated.'


        #calculate final similarity matrix W
        #W = {}
        W_key = 'W'
        for u, related_users in C.items():
            for v, cuv in related_users.items():
                self.my_redis.hset(u, v, C[u][v] / math.sqrt(N[u] * N[v]))
                

    def recommend(self, u, train, N, K=10):
        '''@N: number of user neighbors considered
        '''
        rank = {}
        interacted_items = train[u]

        W_u = self.my_redis.hgetall(u)
        
        for v, wuv in sorted(map(lambda (x, y): (x, float(y)), W_u.items()), key=lambda x: x[1], reverse=True)[0:K]: # wuv: similarity between user u and user v
            for i, rvi in train[v].items(): # rvi: rate of item by user v
                if i in interacted_items:
                    #do not recommend items which user u interacted before
                    continue

                if i not in rank:
                    rank[i] = 0.0
                rank[i] += wuv * rvi

        rank = rank.items()
        rank.sort(key=lambda x: x[1], reverse=True)
        return rank[:N]


class RecommendatorViaWord2Vec(RecommendatorSystemViaCollaborativeFiltering):
    """docstring for RecommendatorViaWord2Vec"""
    def __init__(self):
        super(RecommendatorViaWord2Vec, self).__init__()
        #self.model = None
        self.W = None

    def setup(self, para):
        data = para['data']
        model_name = para['model_name'] if 'model_name' in para else 'tmp_model'
        num_features = para['num_features']
        min_count = para['min_count']
        window = para['window']

        self.K = para['K']

        model_name += ('_'.join(['num_features=' + str(num_features), 'min_count=' + str(min_count), 'window=' + str(window)]) + '.model')

        list_of_list = convert_2_level_dict_to_list_of_list(data)
        #print 'list_of_list:', list_of_list

        print 'start training'
        self.model = gensim.models.Word2Vec(list_of_list, size=num_features, min_count=min_count, window=window)
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
        user_history = {x: data[x].keys() for x in data}
        #print 'user_history:', user_history

        #user repre dict
        user_repre = {uesr_id: np.average(map(lambda item: self.model[item], user_history[uesr_id]), axis=0) for uesr_id in user_history}
        #print 'user_repre:', user_repre

        #
        #calculate final similarity matrix W
        W = {}
        total = len(user_id_set)
        for step, u in enumerate(user_id_set):
            simi_list_of_user_u = []
            for v in user_id_set:
                if u == v:
                    continue

                simi = user_repre[u].dot(user_repre[v]) / (la.norm(user_repre[u] * la.norm(user_repre[v])))
                
                simi_list_of_user_u.append((v, simi))

                #
            K_neighbors = heapq.nlargest(self.K * 2, simi_list_of_user_u, key=lambda s: s[1])
            #print 'K_neighbors:', K_neighbors
            #raw_input()
            W[u] = dict(K_neighbors)
            #print 'W[u]', W[u]
            #raw_input()

            if (0 == step % 64):
                print 'progress: %d/%d' % (step, total)
        print 'progress: %d/%d. done.' % (step, total)
        self.W = W
        


class RecommendatorViaDoc2Vec(RecommendatorSystemViaCollaborativeFiltering):
    """docstring for RecommendatorViaDoc2Vec"""
    def __init__(self):
        super(RecommendatorViaDoc2Vec, self).__init__()
        #self.model = None
        self.W = None

    def setup(self, para):
        data = para['data']
        model_name = para['model_name'] if 'model_name' in para else 'tmp_model'
        num_features = para['num_features']
        min_count = para['min_count']
        window = para['window']

        self.K = para['K']

        model_name += ('_'.join(['num_features=' + str(num_features), 'min_count=' + str(min_count), 'window=' + str(window)]) + '.model')

        list_of_list = convert_2_level_dict_to_list_of_LabeledSentence(data)
        #print 'list_of_list:', list_of_list

        tricky__load_model = False # tricky flag used to skip calculation of model from scratch and load model from file
        if not tricky__load_model:
            print 'start training'
            self.model = gensim.models.Doc2Vec(list_of_list, size=num_features, min_count=min_count, window=window)
            print 'training finished'

            # If you don't plan to train the model any further, calling 
            # init_sims will make the model much more memory-efficient.
            self.model.init_sims(replace=True)

            # It can be helpful to create a meaningful model name and 
            # save the model for later use. You can load it later using Word2Vec.load()
            self.model.save(model_name)
        else:
            self.model = gensim.models.Word2Vec.load('ml-latest-small\\ratings.csv_main_doc2vec_modelnum_features=100_min_count=3_window=20')
            #self.model = gensim.models.Word2Vec.load('ml-latest-small\\ratings.csv_main_doc2vec_modelnum_features=300_min_count=3_window=20')
            

        #
        #user set
        user_id_set = data.keys()

        #user history dict
        user_history = {x: data[x].keys() for x in data}
        #print 'user_history:', user_history

        #user repre dict
        user_repre = {uesr_id: np.average(map(lambda item: self.model[item], filter(lambda x: x in self.model, user_history[uesr_id])), axis=0) for uesr_id in user_history}
        #print 'user_repre:', user_repre

        #
        #calculate final similarity matrix W
        

        W = {}
        total = len(user_id_set)
        for step, u in enumerate(user_id_set):
            simi_list_of_user_u = []
            for v in user_id_set:
                if u == v:
                    continue

                simi = user_repre[u].dot(user_repre[v]) / (la.norm(user_repre[u] * la.norm(user_repre[v])))
                
                simi_list_of_user_u.append((v, simi))

                #
            K_neighbors = heapq.nlargest(self.K * 2, simi_list_of_user_u, key=lambda s: s[1])
            #print 'K_neighbors:', K_neighbors
            #raw_input()
            W[u] = dict(K_neighbors)
            #print 'W[u]', W[u]
            #raw_input()

            if (0 == step % 64):
                print 'progress: %d/%d' % (step, total)
        print 'progress: %d/%d. done.' % (step, total)
        self.W = W
        

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
    }
    print mode_dict
    mode = int(raw_input('Please select mode:'))
    if 1 == mode:
        data_filename, delimiter = os.path.sep.join(['ml-latest-small', 'ratings.csv']), ','

        seed = 2 
        train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 2, 0, seed, delimiter)

        rs = RecommendatorSystemViaCollaborativeFiltering()
        K = 10
        
        for N in xrange(10, 11):
        #for N in xrange(3, 50):
            print 'N:', N


            rs.setup({'train': train, 'K': K})

            recall = rs.recall(train, test, N)
            print 'recall:', recall
            precision = rs.precision(train, test, N)
            print 'precision:', precision
    elif 2 == mode:
        data_filename, delimiter = os.path.sep.join(['ml-latest-small', 'ratings.csv']), ','
        #data_filename, delimiter = os.path.sep.join(['ml-1m', 'ratings.dat']), '::'
        #data_filename, delimiter = os.path.sep.join(['ml-10M100K', 'ratings.dat']), '::'

        K = 10
        seed = 2 
        train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 2, 0, seed, delimiter)

        rs = RecommendatorViaWord2Vec()
        rs.setup({'data': train, 
            'model_name': data_filename + '_' + 'main_word2vec_model',
            'num_features': 100,
            'min_count': 1,
            'window': 5,
            'K': K,
        })

        N = 10
        recall = rs.recall(train, test, N)
        print 'recall:', recall
        precision = rs.precision(train, test, N)
        print 'precision:', precision
    elif 3 == mode:
        data_filename, delimiter = os.path.sep.join(['ml-latest-small', 'ratings.csv']), ','
        #data_filename, delimiter = os.path.sep.join(['ml-1m', 'ratings.dat']), '::'
        #data_filename, delimiter = os.path.sep.join(['ml-10M100K', 'ratings.dat']), '::'

        K = 10
        seed = 2 
        train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 2, 0, seed, delimiter)

        rs = RecommendatorViaDoc2Vec()
        rs.setup({'data': train, 
            'model_name': data_filename + '_' + 'main_doc2vec_model',
            'num_features': 100,
            'min_count': 3,
            'window': 20,
            'K': K,
        })

        N = 10
        recall = rs.recall(train, test, N)
        print 'recall:', recall
        precision = rs.precision(train, test, N)
        print 'precision:', precision



def main_Linux():
    #data_filename, delimiter = os.path.sep.join(['ml-latest-small', 'ratings.csv']), ','
    #data_filename, delimiter = os.path.sep.join(['ml-1m', 'ratings.dat']), '::'
    data_filename, delimiter = os.path.sep.join(['ml-10M100K', 'ratings.dat']), '::'

    seed = 2 
    train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 2, 0, seed, delimiter)

#    #rs = RecommendatorSystemViaCollaborativeFiltering()
#    rs = RecommendatorSystemViaCollaborativeFiltering_UsingRedis()
#
#    rs.setup({'train': train})
#    
#    for N in xrange(10, 11):
#    #for N in xrange(3, 50):
#        print 'N:', N
#
#        recall = rs.recall(train, test, N)
#        print 'recall:', recall
#        precision = rs.precision(train, test, N)
#        print 'precision:', precision

    ###
#    rs = RecommendatorViaWord2Vec()
#    rs.setup({'data': train, 
#        'model_name': 'main_model',
#        'num_features': 100,
#        'min_count': 1,
#        'window': 20,
#    })
#
#    N = 10
#    recall = rs.recall(train, test, N)
#    print 'recall:', recall
#    precision = rs.precision(train, test, N)
#    print 'precision:', precision

    ###
    N = 10
    K = 10

    rs = RecommendatorViaDoc2Vec()
    rs.setup({'data': train, 
        'model_name': data_filename + '_' + 'main_doc2vec_model',
        'num_features': 300,
        'min_count': 3,
        'window': 20,
        'K': K,
    })

    recall = rs.recall(train, test, N)
    print 'recall:', recall
    precision = rs.precision(train, test, N)
    print 'precision:', precision

def main():
    if isWindowsSystem():
        main_windows()
        return

    if isLinuxSystem():
        main_Linux()
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

def test():
    train = {'A': {'a': 1, 'b': 1, 'd': 1},
        'B': {'a': 1, 'c': 1},
        'C': {'b': 1, 'e': 1},
        'D': {'c': 1, 'd': 1, 'e': 1},}

    #test = {'A': {'a': 1, 'b': 1, 'd': 1}, }
    test = {'A': {'c': 1}, }

    #list_of_list = convert_2_level_dict_to_list_of_list(train)
    #print 'list_of_list:', list_of_list

    #rs = RecommendatorViaWord2Vec()
    #rs.setup({'data': train, 'model_name': 'test_model'})


    rs = RecommendatorSystemViaCollaborativeFiltering()
    rs.setup({'train': train, 'K': 10})

    print 'rs.W:'
    print_matrix(rs.W)

    N = 10
    recall = rs.recall(train, test, N)
    print 'recall:', recall
    precision = rs.precision(train, test, N)
    print 'precision:', precision


if __name__ == '__main__':
    main()
    #test()

# -*- coding: utf-8 -*-  

import gensim#from gensim.models import word2vec
import math
import random
import csv
import numpy as np
from numpy import linalg as la
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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
        return hit / (all * 1.0)


    def precision(self, train, test, N):
        hit = 0
        all = 0
        for user in test.keys():
            tu = test[user]
            rank = self.recommend(user, train, N)
            for item, pui in rank:
                if item in tu:
                    hit += 1
            all += len(rank) #Note: In book RSP, the author used 'all += N'
        return hit / (all * 1.0)


class RecommendatorSystemViaCollaborativeFiltering(RecommendatorSystem):
    """docstring for RecommendatorSystemViaCollaborativeFiltering"""
    def __init__(self):
        super(RecommendatorSystemViaCollaborativeFiltering, self).__init__()
        self.W = None   # weight matrix / user similarity matrix

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

        #calculate final similarity matrix W
        W = {}
        for u, related_users in C.items():
            for v, cuv in related_users.items():
                if u not in W:
                    W[u] = {}
                W[u][v] = C[u][v] / math.sqrt(N[u] * N[v])
        self.W = W

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

        model_name += ('_'.join(['num_features=' + str(num_features), 'min_count=' + str(min_count), 'window=' + str(window)]))

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
        for u in user_id_set:
            for v in user_id_set:
                if u == v:
                    continue

                simi = user_repre[u].dot(user_repre[v]) / (la.norm(user_repre[u] * la.norm(user_repre[v])))
                if u not in W:
                    W[u] = {}
                W[u][v] = simi
                if v not in W:
                    W[v] = {}
                W[v][u] = simi
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

        model_name += ('_'.join(['num_features=' + str(num_features), 'min_count=' + str(min_count), 'window=' + str(window)]))

        list_of_list = convert_2_level_dict_to_list_of_LabeledSentence(data)
        #print 'list_of_list:', list_of_list

        print 'start training'
        self.model = gensim.models.Doc2Vec(list_of_list, size=num_features, min_count=min_count, window=window)
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
        user_repre = {uesr_id: np.average(map(lambda item: self.model[item], filter(lambda x: x in self.model, user_history[uesr_id])), axis=0) for uesr_id in user_history}
        #print 'user_repre:', user_repre

        #
        #calculate final similarity matrix W
        W = {}
        for u in user_id_set:
            for v in user_id_set:
                if u == v:
                    continue

                simi = user_repre[u].dot(user_repre[v]) / (la.norm(user_repre[u] * la.norm(user_repre[v])))
                if u not in W:
                    W[u] = {}
                W[u][v] = simi
                if v not in W:
                    W[v] = {}
                W[v][u] = simi
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


def main():
    #data = extract_data()
    #data_filename, delimiter = 'ml-latest-small\\ratings.csv', ','
    data_filename, delimiter = 'ml-1m\\ratings.dat', '::'

    seed = 2 
    train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 2, 0, seed, delimiter)

#    rs = RecommendatorSystemViaCollaborativeFiltering()
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

#    N = 10
#    recall = rs.recall(train, test, N)
#    print 'recall:', recall
#    precision = rs.precision(train, test, N)
#    print 'precision:', precision

    ###
    rs = RecommendatorViaDoc2Vec()
    rs.setup({'data': train, 
        'model_name': data_filename + '_' + 'main_doc2vec_model',
        'num_features': 100,
        'min_count': 3,
        'window': 20,
    })

    N = 10
    recall = rs.recall(train, test, N)
    print 'recall:', recall
    precision = rs.precision(train, test, N)
    print 'precision:', precision

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
    model_name = "300features_40minwords_10context"
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

    rs = RecommendatorViaWord2Vec()
    rs.setup({'data': train, 'model_name': 'test_model'})

    N = 10
    recall = rs.recall(train, test, N)
    print 'recall:', recall
    precision = rs.precision(train, test, N)
    print 'precision:', precision


if __name__ == '__main__':
    main()
    #test()
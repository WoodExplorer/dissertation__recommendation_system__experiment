# -*- coding: utf-8 -*-  

import math
import random
import csv

class RecommendatorSystem(object):
    """docstring for RecommendatorSystem"""
    def __init__(self):
        super(RecommendatorSystem, self).__init__()
        self.W = None   # weight matrix / user similarity matrix
        

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

    def extract_data_from_file_and_generate_train_and_test(self, filename, M, k, seed):
        test = {}
        train = {}
        random.seed(seed)

        with open(filename , 'r') as f:
            first_line = f.readline()
            for i, line in enumerate(f):
                userId, movieId, rating, timestamp = line.split(',')
                userId = int(userId)
                movieId = int(movieId)
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
    filename = 'ml-latest-small\\ratings.csv'

    rs = RecommendatorSystem()
    seed = 2
    
    train, test = rs.extract_data_from_file_and_generate_train_and_test(filename, 2, 0, seed)

    rs.user_similarity(train)
    
    for N in xrange(3, 50):
        print 'N:', N

        recall = rs.recall(train, test, N)
        print 'recall:', recall
        precision = rs.precision(train, test, N)
        print 'precision:', precision


def test():
    train = {'A': {'a': 1, 'b': 1, 'd': 1},
        'B': {'a': 1, 'c': 1},
        'C': {'b': 1, 'e': 1},
        'D': {'c': 1, 'd': 1, 'e': 1},}

    test = {'A': {'a': 1, 'b': 1, 'd': 1}, }
    test = {'A': {'c': 1}, }


    rs = RecommendatorSystem()

    u = 'A'
    N = 10
    recommendation = rs.recommend(u, train, N)
    
    print_matrix(rs.W)

    print 'recommendation:', recommendation
    #print_matrix(recommendation)

    recall = rs.recall(train, test, N)
    print 'recall:', recall
    precision = rs.precision(train, test, N)
    print 'precision:', precision

if __name__ == '__main__':
    main()
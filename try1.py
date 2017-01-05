# -*- coding: utf-8 -*-  

import math
import random

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
        for user, item in data:
            if k == random.randint(0, M):
                test.append([user, item])
            else:
                train.append([user, item])
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

    def recommend(self, u, train, N=10):
        if self.W is None:
            self.user_similarity(train)

        rank = {}
        interacted_items = train[u]
        for v, wuv in sorted(self.W[u].items(), key=lambda x: x[1], reverse=True)[0:N]: # wuv: similarity between user u and user v
            for i, rvi in train[v].items(): # rvi: rate of item by user v
                if i in interacted_items:
                    #do not recommend items which user u interacted before
                    continue

                if i not in rank:
                    rank[i] = 0.0
                rank[i] += wuv * rvi
        return rank


    def recall(self, train, test, N):
        hit = 0
        all = 0
        for user in test.keys():
            tu = test[user]
            rank = self.recommend(user, train, N)
            #print 'rank:', rank
            for item, pui in rank.items():
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
            for item, pui in rank.items():
                if item in tu:
                    hit += 1
            all += len(rank.keys()) #Note: In book RSP, the author used 'all += N'
        return hit / (all * 1.0)


def print_matrix(M):
    def print_wrapper(x):
        print x, M[x]
    map(lambda x: print_wrapper(x), M)


def main():
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
# -*- coding: utf-8 -*-
import random
import math
import numpy as np
import collections as coll

class Conf(object):
    def __init__(self):
        self.time_coef = 0.9

conf = Conf()

def set_time_coef(val):
    global conf
    conf.time_coef = val

def get_time_coef(val):
    global conf
    return conf.time_coef

def get_user_repr_func(ur_name):
    global ur_dict
    return ur_dict[ur_name]

def user_history2user_repr__simple(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = map(lambda x: model[x[0]], items_existed_in_model)
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    #items_multiplied_by_rate = map(lambda (vec, rate, timestamp): vec * rate, items_translated_to_vecs)
    #print 'items_multiplied_by_rate:', items_multiplied_by_rate[0]
    #raw_input()
    
    ## method 1: simple average. not normalized.
    return np.average(items_translated_to_vecs, axis=0)   


def user_history2user_repr__rating(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = np.array(map(lambda x: model[x[0]], items_existed_in_model))
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    
    w = np.array([x[1] for x in items_existed_in_model]) * 1.
    
    return w.dot(items_translated_to_vecs) / np.sum(w)


def user_history2user_repr__simple__time(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    global conf
    time_coef = conf.time_coef

    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = np.array(map(lambda x: model[x[0]], items_existed_in_model))
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    
    #return reduce(lambda x, y: 0.9 * x + 0.1 * y, items_translated_to_vecs)

    w = time_coef ** np.array(range(len(items_translated_to_vecs) - 1, 0 - 1, -1)) * 1.
    
    return w.dot(items_translated_to_vecs) / np.sum(w)

def user_history2user_repr__rating__time(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    global conf
    time_coef = conf.time_coef

    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'num of filtered items:', len(target_user_history) - len(items_existed_in_model)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = np.array(map(lambda x: model[x[0]], items_existed_in_model))
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    
    #return reduce(lambda x, y: 0.9 * x + 0.1 * y, items_translated_to_vecs)
    # Careful! We should use items_existed_in_model rather than target_user_history!!!
    w = time_coef ** np.array(range(len(items_translated_to_vecs) - 1, 0 - 1, -1)) * np.array([x[1] for x in items_existed_in_model]) * 1.
    
    return w.dot(items_translated_to_vecs) / np.sum(w)

### user representation involving tfidf


###
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

tfidf_ready = False
tfdif = None
def init_tfidf(rating_file_name, rating_file_delimiter):
    global tfidf_ready, tfidf

    #rating_file_name, rating_file_delimiter = os.path.sep.join(['ml-1m', 'ratings.dat']), '::'
    user_item_interaction = extract_user_item_interaction(rating_file_name, rating_file_delimiter)

    df = coll.defaultdict(int)
    for user in user_item_interaction:
        distinct_items = set([x[0] for x in user_item_interaction[user]])
        for item in distinct_items:
            df[item] += 1
    num_of_users = len(user_item_interaction.keys())
    
    idf = {item: math.log(num_of_users * 1. / df[item]) for item in df}

    tfidf = idf
    tfidf_ready = True


def assert_tfidf_ready():
    global tfidf_ready
    assert(tfidf_ready)

def user_history2user_repr__simple__tfidf(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    assert_tfidf_ready()
    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = np.array(map(lambda x: model[x[0]], items_existed_in_model))
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    w = np.array([tfidf[x[0]] for x in items_existed_in_model]) * 1.
    
    return w.dot(items_translated_to_vecs) / np.sum(w)   


def user_history2user_repr__rating__tfidf(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    assert_tfidf_ready()
    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = np.array(map(lambda x: (model[x[0]]), items_existed_in_model))
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    
    w = np.array([tfidf[x[0]] for x in items_existed_in_model]) * 1. * np.array([x[1] for x in items_existed_in_model])
    
    return w.dot(items_translated_to_vecs) / np.sum(w)


def user_history2user_repr__simple__time__tfidf(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    assert_tfidf_ready()
    global conf
    time_coef = conf.time_coef

    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = np.array(map(lambda x: model[x[0]], items_existed_in_model))
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    
    #return reduce(lambda x, y: 0.9 * x + 0.1 * y, items_translated_to_vecs)

    w = time_coef ** np.array(range(len(items_translated_to_vecs) - 1, 0 - 1, -1)) * 1. * np.array([tfidf[x[0]] for x in items_existed_in_model])
    
    return w.dot(items_translated_to_vecs) / np.sum(w)

def user_history2user_repr__rating__time__tfidf(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    assert_tfidf_ready()
    global conf
    time_coef = conf.time_coef

    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'num of filtered items:', len(target_user_history) - len(items_existed_in_model)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = np.array(map(lambda x: model[x[0]], items_existed_in_model))
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    
    #return reduce(lambda x, y: 0.9 * x + 0.1 * y, items_translated_to_vecs)
    # Careful! We should use items_existed_in_model rather than target_user_history!!!
    w = time_coef ** np.array(range(len(items_translated_to_vecs) - 1, 0 - 1, -1)) * np.array([x[1] for x in items_existed_in_model]) * 1. * np.array([tfidf[x[0]] for x in items_existed_in_model])
    
    return w.dot(items_translated_to_vecs) / np.sum(w)


ur__simple = 'simple'
ur__rating = 'rating'
ur__simple__time = 'simple_time'
ur__rating__time = 'rating_time'
ur__simple__tfidf = 'simple_tfidf'
ur__rating__tfidf = 'rating_tfidf'
ur__simple__time__tfidf = 'simple_time_tfidf'
ur__rating__time__tfidf = 'rating_time_tfidf'


ur_dict = {
    ur__simple: user_history2user_repr__simple,
    ur__rating: user_history2user_repr__rating,
    ur__simple__time: user_history2user_repr__simple__time,
    ur__rating__time: user_history2user_repr__rating__time,

    ur__simple__tfidf: user_history2user_repr__simple__tfidf,
    ur__rating__tfidf: user_history2user_repr__rating__tfidf,
    ur__simple__time__tfidf: user_history2user_repr__simple__time__tfidf,
    ur__rating__time__tfidf: user_history2user_repr__rating__time__tfidf,
}


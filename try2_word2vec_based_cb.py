# -*- coding: utf-8 -*-  
"""Naval Fate.

Usage:
  naval_fate.py [--cf_on=False]
  naval_fate.py [--compare_variants=False]
  naval_fate.py observe_word2vec_hyperpara (min_count | window)
  naval_fate.py time_overhead
  naval_fate.py time_overhead_CF
  naval_fate.py observe_CF_when_K_varies
  naval_fate.py observe_word2vec_when_K_varies
  naval_fate.py full_comparison

  naval_fate.py ship <name> move <x> <y> [--speed=<kn>]
  naval_fate.py ship shoot <x> <y>
  naval_fate.py mine (set|remove) <x> <y> [--moored | --drifting]
  naval_fate.py (-h | --help)
  naval_fate.py --version

Options:
  -h --help     Show this screen.

"""
from docopt import docopt

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
from utility_extract_data import extract_data_from_file_and_generate_train_and_test
import datetime
import cPickle

def extract_item_info(filename, delimiter, genre_delimiter):
    data = {}

    with open(filename , 'r') as f:
        for i, line in enumerate(f):
            itemId, title, genre_list = map(lambda x: x.strip(), line.split(delimiter))
            
            data[itemId] = (title, genre_list.split(genre_delimiter))
    return data


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


def user_history2user_repr__simple_average(model, target_user_history): # target_user_history: It should_be_a_list_of_tuples_included_items.
    #print 'target_user_history:', target_user_history
    items_existed_in_model = filter(lambda x: x[0] in model, target_user_history)
    #print 'items_existed_in_model:', items_existed_in_model[0]
    items_translated_to_vecs = map(lambda x: model[x[0]], items_existed_in_model)
    #print 'items_translated_to_vecs:', items_translated_to_vecs[0]
    return np.average(items_translated_to_vecs, axis=0)   


def calculate_metrics(test, rec):
    starttime = datetime.datetime.now()
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


def main():
    item_file_name, item_file_delimiter, genre_delimiter = os.path.sep.join(['ml-1m', 'movies.dat']), '::', '|'
    item_info = extract_item_info(item_file_name, item_file_delimiter, genre_delimiter)


    rating_file_name, rating_file_delimiter = os.path.sep.join(['ml-1m', 'ratings.dat']), '::'
    user_item_interaction = extract_user_item_interaction(rating_file_name, rating_file_delimiter)

    model_path = '/home/wsyj/dissertation__recommendation_system__experiment_2/dissertation__recommendation_system__experiment/main_modelnum_features=100_min_count=1_window=1_iter=30.model'

    #model = gensim.models.Word2Vec.load('/home/wsyj/dissertation__recommendation_system__experiment_2/dissertation__recommendation_system__experiment/main_modelnum_features=200_min_count=5_window=2.model' )
    model = gensim.models.Word2Vec.load(model_path)

    #user_history2user_repr__simple_average(model, user_item_interaction['5989'])   

    # calculate user representation dict
    user_repr = {user: user_history2user_repr__simple_average(model, user_item_interaction[user]) 
                 for user in user_item_interaction}

    # item representation
    item_repr = model
    #print len(item_repr)
    
    all_items = set(model.wv.vocab.keys())

    # load train and test datasets

    data_filename, delimiter, data_set = os.path.sep.join(['ml-1m', 'ratings.dat']), '::', '1M'
    #data_filename, delimiter = os.path.sep.join(['ml-10M100K', 'ratings.dat']), '::'
    #data_filename, delimiter, data_set = os.path.sep.join(['ml-100k', 'u.data']), '\t', '100K'

    N = 20
    seed = 2 
    K = 10
    train_percent = 0.8
    test_data_inner_ratio = 0.8
    test = None
    train, original_test = extract_data_from_file_and_generate_train_and_test(data_filename, train_percent, seed, delimiter, test_data_inner_ratio)
    #train, test = extract_data_from_file_and_generate_train_and_test(data_filename, 3, 0, seed, delimiter)


    # main: core of content-based recommendation

    total_user_item_comb = 0
    rec = {}
    for user in original_test:
        history, future = original_test[user]
        history_items = set([x[0] for x in history])
        candidates = all_items - history_items # filtering out those interacted
        #print 'candidates:', candidates

        total_user_item_comb += len(candidates)
        cand_simi_list = []
        for candy in candidates:
            simi = user_repr[user].dot(item_repr[candy]) / (la.norm(user_repr[user]) * la.norm(item_repr[candy]))
            cand_simi_list.append((candy, simi))

        cand_simi_list.sort(key=lambda x: -1 * x[1])

        rec[user] = cand_simi_list[:N]

    print 'total_user_item_comb:', total_user_item_comb

    metrics = calculate_metrics(original_test, rec)
    cPickle.dump(rec, open('try2.output.dmp', 'w'))
    print "metrics:", metrics



if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1.1rc')
    #print(arguments)
    #print arguments['--cf_on']
    #exit(0)
    #print(arguments)
    #exit(0)
    main()
    #test()

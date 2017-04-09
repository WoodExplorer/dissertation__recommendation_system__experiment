# -*- coding: utf-8 -*-
import random

def extract_data_from_file_and_generate_train_and_test(filename, train_percent, seed, delimiter, test_data_inner_ratio, sort_by_time=False, test_fixed_ratio=None):
    test = None
    train = None
    data = {}
    random.seed(seed)

    total_users = None
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
    total_users = len(data.keys())

    test = {}
    train = {}
    for userId in data:
        total_len = len(data[userId])
        if random.random() >= train_percent:
        #if 0 == random.randint(0, 2):
        #if 2 == random.randint(0, 2):
            add_it = True
            if test_fixed_ratio is None:
                pass
            else:
                if len(test) >= (train_percent * total_users * test_fixed_ratio):
                    add_it = False
            if add_it:    
                test[userId] = data[userId]
        else:
            train[userId] = data[userId]

    print 'len(train):', len(train)
    print 'len(test):', len(test)

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


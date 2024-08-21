import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

# def build_index(dataset_name):

#     ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

#     n_users = ui_mat[:, 0].max()
#     n_items = ui_mat[:, 1].max()

#     u2i_index = [[] for _ in range(n_users + 1)]
#     i2u_index = [[] for _ in range(n_items + 1)]

#     for ui_pair in ui_mat:
#         u2i_index[ui_pair[0]].append(ui_pair[1])
#         i2u_index[ui_pair[1]].append(ui_pair[0])

#     return u2i_index, i2u_index


# def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
#     def sample(uid):

#         # uid = np.random.randint(1, usernum + 1)
#         while len(user_train[uid]) <= 1: user = np.random.randint(1, usernum + 1)

#         seq = np.zeros([maxlen], dtype=np.int32)
#         pos = np.zeros([maxlen], dtype=np.int32)
#         neg = np.zeros([maxlen], dtype=np.int32)
#         nxt = user_train[uid][-1]
#         idx = maxlen - 1

#         ts = set(user_train[uid])
#         for i in reversed(user_train[uid][:-1]):
#             seq[idx] = i
#             pos[idx] = nxt
#             if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
#             nxt = i
#             idx -= 1
#             if idx == -1: break

#         return (uid, seq, pos, neg)

#     np.random.seed(SEED)
#     uids = np.arange(1, usernum+1, dtype=np.int32)
#     counter = 0
#     while True:
#         if counter % usernum == 0:
#             np.random.shuffle(uids)
#         one_batch = []
#         for i in range(batch_size):
#             one_batch.append(sample(uids[counter % usernum]))
#             counter += 1
#         result_queue.put(zip(*one_batch))


# class WarpSampler(object):
#     def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
#         self.result_queue = Queue(maxsize=n_workers * 10)
#         self.processors = []
#         for i in range(n_workers):
#             self.processors.append(
#                 Process(target=sample_function, args=(User,
#                                                       usernum,
#                                                       itemnum,
#                                                       batch_size,
#                                                       maxlen,
#                                                       self.result_queue,
#                                                       np.random.randint(2e9)
#                                                       )))
#             self.processors[-1].daemon = True
#             self.processors[-1].start()

#     def next_batch(self):
#         return self.result_queue.get()

#     def close(self):
#         for p in self.processors:
#             p.terminate()
#             p.join()


# train/val/test data generation
# def data_partition(fname):
#     usernum = 0
#     itemnum = 0
#     User = defaultdict(list)
#     user_train = {}
#     user_valid = {}
#     user_test = {}
#     # assume user/item index starting from 1
#     f = open('data/%s.txt' % fname, 'r')
#     for line in f:
#         u, i = line.rstrip().split(' ')
#         u = int(u)
#         i = int(i)
#         usernum = max(u, usernum)
#         itemnum = max(i, itemnum)
#         User[u].append(i)

#     for user in User:
#         nfeedback = len(User[user])
#         if nfeedback < 3:
#             user_train[user] = User[user]
#             user_valid[user] = []
#             user_test[user] = []
#         else:
#             user_train[user] = User[user][:-2]
#             user_valid[user] = []
#             user_valid[user].append(User[user][-2])
#             user_test[user] = []
#             user_test[user].append(User[user][-1])
#     return [user_train, user_valid, user_test, usernum, itemnum]


def build_index(dataset_name):
    u2i_index = defaultdict(list)
    i2u_index = defaultdict(list)
    usernum = 0
    itemnum = 0
    with open(f'data/{dataset_name}.txt', 'r') as f:
        for line in f:
            u, i = map(int, line.strip().split())
            u2i_index[u].append(i)
            i2u_index[i].append(u)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
    return u2i_index, i2u_index, usernum, itemnum



# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):
        user_data = user_train[uid]

        while len(user_data) <= 1:
            uid = np.random.randint(1, usernum + 1)
            user_data = user_train[uid]

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_data[-1]
        idx = maxlen - 1

        ts = set(user_data)
        for i in reversed(user_data[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, user_train, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(user_train,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def data_partition(fname):
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)
    usernum = 0
    itemnum = 0

    with open(f'data/{fname}.txt', 'r') as f:
        for line in f:
            u, i = map(int, line.strip().split())
            user_train[u].append(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)

    # Partitioning the data
    for user in user_train:
        nfeedback = len(user_train[user])
        if nfeedback < 3:
            user_valid[user] = []
            user_test[user] = []
        else:
            user_valid[user] = [user_train[user][-2]]
            user_test[user] = [user_train[user][-1]]
            user_train[user] = user_train[user][:-2]

    return user_train, user_valid, user_test, usernum, itemnum

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

import sys
import copy
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):

    ui_mat = np.loadtxt(f'data/{dataset_name}/dataset/{dataset_name}.txt', dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    uids = np.arange(1, usernum+1, dtype=np.int32)
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
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(f'data/{fname}/dataset/{fname}.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

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
        for rand_item in range(1, itemnum + 1):
            if rand_item in rated:
                continue
            else:
                item_idx.append(rand_item)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


# def evaluate_split(model, dataset, args, user_groups,  split = None):

#     alltypes = sorted(np.unique(np.array(list(user_groups.values()))).tolist(), key=lambda x: int(x.split('_')[1])) ### usersplit group ### usersplit group
#     ndcg = dict(zip(alltypes,np.zeros(len(alltypes))))
#     ht = dict(zip(alltypes,np.zeros(len(alltypes))))
#     valid_count = dict(zip(alltypes,np.zeros(len(alltypes))))
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     users = range(1, usernum + 1)
#     for u in users:
#         usertype = user_groups[str(u)]
#         if split != None and usertype != split:
#             continue
#         if len(train[u]) < 1 or len(test[u]) < 1: 
#             continue
#         valid_count[usertype] += 1
#         seq = np.zeros([args.maxlen], dtype=np.int32) ### max len
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [test[u][0]] ### true label

#         for rand_item in range(1, itemnum + 1):
#             if rand_item in rated:
#                 continue
#             else:
#                 item_idx.append(rand_item)


#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions = predictions[0] # - for 1st argsort DESC

#         rank = predictions.argsort().argsort()[0].item()

#         if rank < 10:
#             ndcg[usertype] += 1 / np.log2(rank + 2)
#             ht[usertype] += 1

#     if split == None:
#         ndcg_all = dict()
#         ht_all = dict()
#         for i in alltypes:
#             ndcg_all[i] = ndcg[i] / valid_count[i]
#             ht_all[i] = ht[i] / valid_count[i]
        
#         return ndcg_all, ht_all
#     else:
#         return ndcg[split]/valid_count[split], ht[split]/valid_count[split]


# def count_trainable_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def evaluate_user(model, dataset, args, test_user):

#     ndcg = 0
#     ht = 0
#     valid_count = 0
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000) ### randomly select 10000 users for validation
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if u!= test_user:
#             continue
#         if len(train[u]) < 1 or len(test[u]) < 1: 
#             continue
#         valid_count += 1
#         seq = np.zeros([args.maxlen], dtype=np.int32) ### max len
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [test[u][0]] ### true label

#         for rand_item in range(1, itemnum + 1):
#             if rand_item in rated:
#                 continue
#             else:
#                 item_idx.append(rand_item)


#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions = predictions[0] # - for 1st argsort DESC

#         rank = predictions.argsort().argsort()[0].item()

#         if rank < 10:
#             ndcg += 1 / np.log2(rank + 2)
#             ht += 1

#     return ndcg/valid_count, ht/valid_count




# def evaluate_user_multiple_model(models:list, dataset, args, test_user, alpha):

#     ndcg = 0
#     ht = 0
#     valid_count = 0
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000) ### randomly select 10000 users for validation
#     else:
#         users = range(1, usernum + 1)
#     for u in users:
#         if u!= test_user:
#             continue
#         if len(train[u]) < 1 or len(test[u]) < 1: 
#             continue
#         valid_count += 1
#         seq = np.zeros([args.maxlen], dtype=np.int32) ### max len
#         idx = args.maxlen - 1
#         seq[idx] = valid[u][0]
#         idx -= 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break
#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [test[u][0]] ### true label

#         for rand_item in range(1, itemnum + 1):
#             if rand_item in rated:
#                 continue
#             else:
#                 item_idx.append(rand_item)


#         features = []
#         item_embeddings = []
#         for model in models:
#             final_feat, item_embeds = model.predict_helper(*[np.array(l) for l in [[u], [seq], item_idx]])
#             features.append(final_feat)
#             item_embeddings.append(item_embeds)
        
#         combined_log_feats = alpha * features[0] + (1-alpha) * features[1]
#         item_embedding = item_embeddings[0]
#         predictions = -item_embedding.matmul(combined_log_feats.unsqueeze(-1)).squeeze(-1)
#         predictions = predictions[0] # - for 1st argsort DESC

#         rank = predictions.argsort().argsort()[0].item()

#         if rank < 10:
#             ndcg += 1 / np.log2(rank + 2)
#             ht += 1

#     return ndcg/valid_count, ht/valid_count
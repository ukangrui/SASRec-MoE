import pandas as pd
import json
import numpy as np
import random
import copy

def load_dataset(data_dir):
    users = []
    items = []
    f = open(data_dir, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        users.append(u)
        items.append(i)
    users = pd.Series(users)
    items = pd.Series(items)
    df = pd.DataFrame({'user': users, 'item': items})
    return df

def split_by(df: pd.DataFrame, mode, num_groups = 100):
    num_users  = len(df['user'].unique())
    num_items  = len(df['item'].unique())
    popular_items = df['item'].value_counts(ascending=False).index.to_list()[:num_items//5] ## top 20%
    def filter_helper(batch):
        all = len(batch)
        popular = len(batch[batch['item'].isin(popular_items)])
        return popular / all
    if mode == 'popularity':
        user_sorted = df.groupby('user').apply(filter_helper).sort_values(ascending = False).index.tolist()
    elif mode == 'temprature':
        user_sorted = df.groupby('user')['item'].agg('count').sort_values(ascending=False).index.tolist()
    else:
        raise AssertionError
    cuts  = (np.arange(1,num_groups) * num_users * (1/num_groups)).astype(int) ### 99 cuts
    user_groups = list()
    user_groups.append(user_sorted[:cuts[0]])
    for index, cut in enumerate(cuts):
        if index == num_groups - 2:
            user_groups.append(user_sorted[cut:])
        else:
            user_groups.append(user_sorted[cut:cuts[index+1]])
    return user_groups

def split_users(dataset , mode):
    df = load_dataset(data_dir=f"data/{dataset}/dataset/{dataset}.txt")
    user_groups = split_by(df, mode=mode)
    user_types = {}
    for index, user_group in enumerate(user_groups):
        for user in user_group:
            user_types[user] = "split_" + str(index+1)
    json_data = json.dumps(user_types)
    with open(f'data/{dataset}/lora/{mode}/{mode}.json', 'w+') as json_file:
        json_file.write(json_data)
    print("Finished spliting User Groups:  "+ dataset + ' -- ' + mode)


def evaluate_split(model, dataset, args, user_groups,  split = None):

    alltypes = sorted(np.unique(np.array(list(user_groups.values()))).tolist(), key=lambda x: int(x.split('_')[1])) ### usersplit group ### usersplit group
    ndcg = dict(zip(alltypes,np.zeros(len(alltypes))))
    ht = dict(zip(alltypes,np.zeros(len(alltypes))))
    valid_count = dict(zip(alltypes,np.zeros(len(alltypes))))
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    users = range(1, usernum + 1)
    for u in users:
        usertype = user_groups[str(u)]
        if split != None and usertype != split:
            continue
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue
        valid_count[usertype] += 1
        seq = np.zeros([args.maxlen], dtype=np.int32) ### max len
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]] ### true label

        for rand_item in range(1, itemnum + 1):
            if rand_item in rated:
                continue
            else:
                item_idx.append(rand_item)


        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        if rank < 10:
            ndcg[usertype] += 1 / np.log2(rank + 2)
            ht[usertype] += 1

    if split == None:
        ndcg_all = dict()
        ht_all = dict()
        for i in alltypes:
            ndcg_all[i] = ndcg[i] / valid_count[i]
            ht_all[i] = ht[i] / valid_count[i]
        
        return ndcg_all, ht_all
    else:
        return ndcg[split]/valid_count[split], ht[split]/valid_count[split]


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_user(model, dataset, args, test_user):

    ndcg = 0
    ht = 0
    valid_count = 0
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000) ### randomly select 10000 users for validation
    else:
        users = range(1, usernum + 1)
    for u in users:
        if u!= test_user:
            continue
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue
        valid_count += 1
        seq = np.zeros([args.maxlen], dtype=np.int32) ### max len
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]] ### true label

        for rand_item in range(1, itemnum + 1):
            if rand_item in rated:
                continue
            else:
                item_idx.append(rand_item)


        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        if rank < 10:
            ndcg += 1 / np.log2(rank + 2)
            ht += 1

    return ndcg/valid_count, ht/valid_count


def evaluate_ensemble(models:list, dataset, args, alpha, mode):

    ndcg = 0
    ht = 0
    valid_count = 0
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue
        valid_count += 1
        seq = np.zeros([args.maxlen], dtype=np.int32) ### max len
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]] ### true label

        for rand_item in range(1, itemnum + 1):
            if rand_item in rated:
                continue
            else:
                item_idx.append(rand_item)


        features = []
        item_embeddings = []
        for model in models:
            final_feat, item_embeds = model.predict_helper(*[np.array(l) for l in [[u], [seq], item_idx]])
            features.append(final_feat)
            item_embeddings.append(item_embeds)
        
        if mode == 'MoE':
            combined_log_feats = alpha * features[0] + (1-alpha) * features[1]
            item_embedding = item_embeddings[0]
            predictions = -item_embedding.matmul(combined_log_feats.unsqueeze(-1)).squeeze(-1)
            predictions = predictions[0] # - for 1st argsort DESC
        elif mode == 'Voting':
            pass
        else:
            raise AssertionError

        rank = predictions.argsort().argsort()[0].item()
        if rank < 10:
            ndcg += 1 / np.log2(rank + 2)
            ht += 1
    return ndcg/valid_count, ht/valid_count

def clear_lora_cache():
    pass


def mark_only_gating_trainable(model):
        for name, param in model.named_parameters():
            if "gate" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for name, param in model.named_parameters():
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
    

def dict2json(dict, dir):
    pass
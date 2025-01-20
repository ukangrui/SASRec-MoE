import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data.dataloader import DataLoader

def load_txt_file(ds_name = 'ml-1m'):
    f = open('data/%s.txt' % ds_name, 'r')
    ds_dict = {}
    for time_idx, line in enumerate(f): ### data is already sorted by timestamp
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        ds_dict[u] = ds_dict.get(u, []) + [i]
    return ds_dict

def get_usr_itm_num(ds_name = 'ml-1m'):
    users = []
    items = []
    f = open('data/%s.txt' % ds_name, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        users.append(int(u))
        items.append(int(i))
    users = pd.Series(users)
    items = pd.Series(items)
    return len(users.unique()), len(items.unique())

def pad_seq(seq, maxlen = 200):
    if len(seq) < maxlen:
        seq = [0] * (maxlen - len(seq)) + seq
    else:
        seq = seq[-maxlen:]
    return seq

def random_neg(itemnum, pos):
    neg = np.random.randint(1, itemnum + 1)
    while neg in pos:
        neg = np.random.randint(1, itemnum + 1)
    return neg

def load_train_test_data_per(ds_dict,itemnum, max_len = 200, skip_short = 40, test_per = 0.05):
    train_data = []
    test_data  = []
    for u in tqdm(ds_dict, desc = 'Processing Users'): # Iterate over users
        if len(ds_dict[u]) < skip_short:
            continue
        if len(ds_dict[u]) > max_len:
            ds_dict[u] = ds_dict[u][-max_len:]
        start_test_idx = int(len(ds_dict[u]) * (1 - test_per))
        train_items = ds_dict[u][:start_test_idx]
        test_items =  ds_dict[u][start_test_idx:]
        ### Gather training data
        train_seq = pad_seq(train_items, max_len)
        train_pos = pad_seq(train_items[1:] + [test_items[0]], max_len)
        train_neg = pad_seq([random_neg(itemnum, set(train_items)) for _ in range(len(train_items))], max_len)
        train_data.append((u, train_seq, train_pos, train_neg))
        ### Gather test data
        for i in range(len(test_items)):
            test_seq = pad_seq(train_items + test_items[:i], max_len)
            test_pos = test_items[i]
            test_idxs = [item for item in list(range(1, itemnum + 1)) if item not in test_seq] + [test_pos]
            num_padding = itemnum - len(test_idxs)
            test_idxs = [0] * num_padding + test_idxs
            mask = num_padding
            test_data.append((u, test_seq, test_pos, test_idxs, mask))
    return train_data, test_data


def load_train_test_data_num(ds_dict, itemnum,  max_len = 200, skip_short = 20, num_test = 5):
    train_data = []
    test_data  = []
    for u in tqdm(ds_dict, desc = 'Processing Users'): # Iterate over users
        if len(ds_dict[u]) < skip_short:
            continue
        if len(ds_dict[u]) > max_len:
            ds_dict[u] = ds_dict[u][-max_len:]
        
        train_items = ds_dict[u][:-num_test]
        test_items =  ds_dict[u][-num_test:]
        ### Gather training data
        train_seq = pad_seq(train_items, max_len)
        train_pos = pad_seq(train_items[1:] + [test_items[0]], max_len)
        train_neg = pad_seq([random_neg(itemnum, set(train_items)) for _ in range(len(train_items))], max_len)
        train_data.append((u, train_seq, train_pos, train_neg))
        ### Gather test data
        for i in range(len(test_items)):
            test_seq = pad_seq(train_items + test_items[:i], max_len)
            test_pos = test_items[i]
            test_idxs = [item for item in list(range(1, itemnum + 1)) if item not in test_seq] + [test_pos]
            num_padding = itemnum - len(test_idxs)
            test_idxs = [0] * num_padding + test_idxs
            mask = num_padding
            test_data.append((u, test_seq, test_pos, test_idxs, mask))
    return train_data, test_data


def collate_train(batch):
    u, seq, pos, neg = zip(*batch)
    return torch.LongTensor(u), torch.LongTensor(seq), torch.LongTensor(pos), torch.LongTensor(neg)
def collate_test(batch):
    u, seq, pos, idx, mask = zip(*batch)
    return torch.LongTensor(u), torch.LongTensor(seq), torch.LongTensor(pos), torch.LongTensor(idx), torch.LongTensor(mask)

def eval_step(model, u, seq, pos,test_items, mask):
    ndcg, ht = 0,0
    predictions = -model.predict_batch(u, seq, test_items)
    for (usr_idx, pred,start_idx, test_idxs, label) in zip(u, predictions, mask, test_items, pos):
        pred = pred[start_idx:].detach().cpu()
        test_idxs = test_idxs[start_idx:]
        top_idx = np.argsort(pred)[:10]
        top_items = [int(test_idxs[idx].detach().item()) for idx in top_idx]
        label = int(label.detach().item())
        if label in top_items:
            ndcg += 1 / np.log2(top_items.index(label) + 2)
            ht += 1
    return ndcg, ht


def train_step(model, u, seq, pos, neg, criterion, optimizer):
    pos_logits, neg_logits = model(u, seq, pos, neg)
    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=torch.device('cuda')), torch.zeros(neg_logits.shape, device=torch.device('cuda'))
    optimizer.zero_grad()
    indices = np.where(pos != 0)
    loss = criterion(pos_logits[indices], pos_labels[indices]) + criterion(neg_logits[indices], neg_labels[indices])
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def get_lora_train_test_ds(split_file, train, test, split = 'split_1'):
    lora_users = [int(user) for user in split_file if split_file[user] == split]
    lora_train = [row for row in train if row[0] in lora_users]
    lora_test = [row for row in test if row[0] in lora_users]
    return lora_train, lora_test
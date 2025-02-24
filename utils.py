import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from models.sasrec_moe_soft import *

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


def load_train_test_data_num(ds_dict, itemnum,  max_len = 200, num_test = 1, skip_short = 0):
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

def eval_step(model, u, seq, pos,test_items, mask, topK = 10):
    ndcg, ht = 0,0
    predictions = -model.predict_batch(u, seq, test_items)
    for (usr_idx, pred,start_idx, test_idxs, label) in zip(u, predictions, mask, test_items, pos):
        pred = pred[start_idx:].detach().cpu()
        test_idxs = test_idxs[start_idx:]
        top_idx = np.argsort(pred)[:topK]
        top_items = [int(test_idxs[idx].detach().item()) for idx in top_idx]
        label = int(label.detach().item())
        if label in top_items:
            ndcg += 1 / np.log2(top_items.index(label) + 2)
            ht += 1
    return ndcg, ht


def train_step(model, u, seq, pos, neg, criterion, optimizer, device):
    pos_logits, neg_logits = model(u, seq, pos, neg)
    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=torch.device(device)), torch.zeros(neg_logits.shape, device=torch.device(device))
    optimizer.zero_grad()
    indices = np.where(pos != 0)
    loss = criterion(pos_logits[indices], pos_labels[indices]) + criterion(neg_logits[indices], neg_labels[indices])
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def get_lora_train_test_ds(split_file, train, test, split):
    lora_users = [int(user) for user in split_file if split_file[user] == split]
    lora_train = [row for row in train if row[0] in lora_users]
    lora_test = [row for row in test if row[0] in lora_users]
    return lora_train, lora_test

### 
def load_train_valid_test_data_num(ds_dict, itemnum,  max_len = 200, skip_short = 0, num_valid = 1, num_test = 1):
    train_data = []
    valid_data = []
    test_data  = []
    for u in tqdm(ds_dict, desc = 'Processing Users'): # Iterate over users
        if len(ds_dict[u]) < skip_short:
            continue
        if len(ds_dict[u]) > max_len:
            ds_dict[u] = ds_dict[u][-max_len:]
        
        train_items = ds_dict[u][:-num_test-num_valid]
        valid_items = ds_dict[u][-num_test-num_valid:-num_test]
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
        
        ### Gather valid data
        for i in range(len(valid_items)):
            valid_seq = pad_seq(train_items + valid_items[:i], max_len)
            valid_pos = valid_items[i]
            valid_idxs = [item for item in list(range(1, itemnum + 1)) if item not in valid_seq] + [valid_pos]
            num_padding = itemnum - len(valid_idxs)
            valid_idxs = [0] * num_padding + valid_idxs
            mask = num_padding
            valid_data.append((u, valid_seq, valid_pos, valid_idxs, mask))
        
    return train_data, valid_data, test_data

def collate_valid(batch):
    u, seq, pos, idx, mask = zip(*batch)
    return torch.LongTensor(u), torch.LongTensor(seq), torch.LongTensor(pos), torch.LongTensor(idx), torch.LongTensor(mask)


def get_lora_train_valid_test_ds(split_file, train, valid, test, split = 'split_1'):
    lora_users = [int(user) for user in split_file if split_file[user] == split]
    lora_train = [row for row in train if row[0] in lora_users]
    lora_valid = [row for row in valid if row[0] in lora_users]
    lora_test = [row for row in test if row[0] in lora_users]
    return lora_train, lora_valid, lora_test





def pretrain_router(base_model: SASRecMoESoft, router_model: Router, train_loader, grouping_file, device):
    optimizer = torch.optim.AdamW(router_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    for _ in tqdm(range(20), desc='pretraining router'):
        for train_batch in train_loader:
            router_model.train()
            u, log_seqs, _ , _ = train_batch
            with torch.no_grad():
                seqs = base_model.item_emb(torch.LongTensor(log_seqs).to(device))
                seqs *= base_model.item_emb.embedding_dim ** 0.5
                seqs = torch.mean(seqs, dim=1)
                labels = torch.tensor([int(grouping_file[str(user.item())].split('_')[1]) - 1 for user in u]).to(device)
            optimizer.zero_grad()
            output = router_model(seqs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return router_model
            
def create_target_distribution(n, length=10, beta = 0.9):
    x = torch.arange(1, length + 1).float()
    lambda_param =  beta * ((1 + (length - n) / length) ** 2)
    logits = -lambda_param * torch.abs(x - n)
    probs = F.softmax(logits, dim=0)
    return probs


def pretrain_router_w_distribution(base_model: SASRecMoESoft, router_model: Router, train_loader, grouping_file, device):
    optimizer = torch.optim.AdamW(router_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.MSELoss(reduction='sum')
    for _ in tqdm(range(20), desc='pretraining router'):
        for train_batch in train_loader:
            router_model.train()
            u, log_seqs, _ , _ = train_batch
            with torch.no_grad():
                seqs = base_model.item_emb(torch.LongTensor(log_seqs).to(device))
                seqs *= base_model.item_emb.embedding_dim ** 0.5
                seqs = torch.mean(seqs, dim=1)
                labels = torch.vstack([create_target_distribution(int(grouping_file[str(user.item())].split('_')[1]))  for user in u]).to(device)
            optimizer.zero_grad()
            output = router_model(seqs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return router_model
            

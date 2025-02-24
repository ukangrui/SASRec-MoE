import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models.sasrec_base import SASRec
from utils import *
from tqdm import tqdm
import wandb
import argparse
import wandb
import pickle
import time
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="ml-1m")
parser.add_argument("--max_len", type=int, default=200)
parser.add_argument("--num_holdout", type=int, default=1)
parser.add_argument("--topK", type=int, default=10)
parser.add_argument('--device', type=str, default='mps')
args = parser.parse_args()


num_u, num_i = get_usr_itm_num(args.dataset)


train, test = load_train_test_data_num(load_txt_file(args.dataset), num_i, max_len = args.max_len, num_test=args.num_holdout)
# train = pickle.load(open("data/train_holdout4.pkl", "rb"))
# test = pickle.load(open("data/test_holdout4.pkl", "rb"))


train_loader = DataLoader(train, batch_size = 256, shuffle = True, collate_fn = collate_train)
test_loader  = DataLoader(test, batch_size = 256, shuffle = False, collate_fn = collate_test)
train_loader = DataLoader(train, batch_size = 256, shuffle = True, collate_fn = collate_train)
test_loader  = DataLoader(test, batch_size = 256, shuffle = False, collate_fn = collate_test)

model = SASRec(user_num = num_u, item_num = num_i, maxlen = 200, num_blocks = 2, num_heads = 1, hidden_units = 50, dropout_rate = 0.2, device = args.device)
model.load_state_dict(torch.load(f'checkpoints/{args.dataset}-base.pth', map_location=torch.device(args.device)))
model = model.to(args.device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=1e-4)
# for epoch in range(10):
#     running_loss = 0.0
#     model.train()
#     for train_batch in train_loader:
#         u, seq, pos, neg = train_batch
#         running_loss += train_step(model, u, seq, pos, neg, criterion, optimizer, args.device)
#     print('Epoch', epoch, 'Loss', round(running_loss/len(train),4), end=' ')
    

model.eval()
ndcg, ht = 0, 0
with torch.no_grad():
    for test_batch in test_loader:
        u, seq, pos, test_items, mask = test_batch
        batch_ndcg, batch_ht = eval_step(model, u, seq, pos, test_items, mask, topK = args.topK)
        ndcg += batch_ndcg
        ht += batch_ht
ndcg /= len(test)
ht /= len(test)
print(f'test ndcg: {ndcg}, ht: { ht}')


# torch.save(model.state_dict(), f'checkpoints/{args.dataset}-base{ht}.pth')


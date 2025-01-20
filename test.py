import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models.sasrec_base import SASRec
from utils import *


DATASET  = 'Video'
num_u, num_i = get_usr_itm_num(DATASET)
train, test = load_train_test_data_num(load_txt_file(DATASET), num_i)
train_loader = DataLoader(train, batch_size = 64, shuffle = True, collate_fn = collate_train)
test_loader  = DataLoader(test, batch_size = 64, shuffle = False, collate_fn = collate_test)

model = SASRec(user_num = num_u, item_num = num_i, maxlen = 200, num_blocks = 2, num_heads = 1, hidden_units = 50, dropout_rate = 0.2)
model = model.to('cuda')
model.eval()
ndcg = 0
ht = 0
with torch.no_grad():
    for test_batch in test_loader:
        u, seq, pos, test_items, mask = test_batch
        batch_ndcg, batch_ht = eval_step(model, u, seq, pos, test_items, mask)
        ndcg += batch_ndcg
        ht += batch_ht
print(f'ndcg: {ndcg / len(test)}, ht: { ht / len(test)}')
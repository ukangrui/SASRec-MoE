import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models.sasrec_base import SASRec
from utils import *
from tqdm import tqdm
import wandb
wandb.init(project="moelora",name=f"train_base")


DATASET = 'ml-1m'
num_u, num_i = get_usr_itm_num(DATASET)
train, test = load_train_test_data_num(load_txt_file(DATASET), num_i)
train_loader = DataLoader(train, batch_size = 256, shuffle = True, collate_fn = collate_train)
test_loader  = DataLoader(test, batch_size = 256, shuffle = False, collate_fn = collate_test)

model = SASRec(user_num = num_u, item_num = num_i, maxlen = 200, num_blocks = 2, num_heads = 1, hidden_units = 50, dropout_rate = 0.2)
model.load_state_dict(torch.load('checkpoints/ml-1m-base.pth', map_location=torch.device('cuda')))
model = model.to('cuda')
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, weight_decay=1e-4)
for epoch in range(50):
    running_loss = 0.0
    model.train()
    for train_batch in train_loader:
        u, seq, pos, neg = train_batch
        running_loss += train_step(model, u, seq, pos, neg, criterion, optimizer)
    print('Epoch', epoch, 'Loss', round(running_loss/len(train),4), end=' ')
    wandb.log({"epoch": epoch, "train_loss": round(running_loss/len(train),4)})
    
    model.eval()
    ndcg, ht = 0, 0
    with torch.no_grad():
        for test_batch in test_loader:
            u, seq, pos, test_items, mask = test_batch
            batch_ndcg, batch_ht = eval_step(model, u, seq, pos, test_items, mask)
            ndcg += batch_ndcg
            ht += batch_ht
    print(f'ndcg: {ndcg / len(test)}, ht: { ht / len(test)}')
    wandb.log({"epoch": epoch, "ndcg": float(ndcg / len(test))})
    wandb.log({"epoch": epoch, "ht": float(ht / len(test))})
    torch.save(model.state_dict(), f'checkpoints/{DATASET}-{round(ndcg/len(test),4)}.pth')   
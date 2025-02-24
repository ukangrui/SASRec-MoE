import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models.sasrec_base import SASRec
import json
from utils import *
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
import pickle
import os
parser = argparse.ArgumentParser()

parser.add_argument('--grouping_metric', type=str, default='temperature', choices=['popularity', 'temperature'])

parser.add_argument("--dataset", type=str, default="ml-1m")
parser.add_argument("--max_len", type=int, default=200)
parser.add_argument("--num_holdout", type=int, default=1)
parser.add_argument("--topK", type=int, default=10)
parser.add_argument('--device', type=str, default='mps')

args = parser.parse_args()

print(args.grouping_metric)


with open(f'config/{args.grouping_metric}.json') as f:
    split_file = dict(json.load(f))
all_splits = list(set(split_file.values()))


num_u, num_i = get_usr_itm_num(args.dataset)
# train, test = load_train_test_data_num(load_txt_file(args.dataset), num_i, max_len = args.max_len, num_test=args.num_holdout,)
train = pickle.load(open("data/train_holdout4.pkl", "rb"))
test = pickle.load(open("data/test_holdout4.pkl", "rb"))

full_ndcg, full_ht = 0,0
full_len = 0

for split in all_splits:
    os.makedirs(f'lora_checkpoints/{args.grouping_metric}/{split}', exist_ok=True)
    print(f'training on {split}')

    lora_train, lora_test = get_lora_train_test_ds(split_file, train, test, split)

    train_loader = DataLoader(lora_train, batch_size = 384, shuffle = True, collate_fn = collate_train)
    test_loader  = DataLoader(lora_test, batch_size = 384, shuffle = False, collate_fn = collate_test)


    base_model = SASRec(user_num = num_u, item_num = num_i, maxlen = 200, num_blocks = 2, num_heads = 1, hidden_units = 50, dropout_rate = 0.2, device = args.device)
    base_model.load_state_dict(torch.load(f'checkpoints/ml-1m-base.pth', map_location=torch.device(args.device)))

            
    lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=['attention_layers.0', 'attention_layers.1'],
    bias="none",
)

    model = get_peft_model(base_model, lora_config)


    model = model.to(args.device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    base_ndcg , base_ht = 0, 0
    model.eval()
    with torch.no_grad():
        for test_batch in test_loader: 
            u, seq, pos, test_items, mask = test_batch
            batch_ndcg, batch_ht = eval_step(model, u, seq, pos, test_items, mask)
            base_ndcg += batch_ndcg
            base_ht += batch_ht
    base_ndcg /= len(lora_test)
    base_ht /= len(lora_test)
    # print(f'split{split},  base ndcg: {base_ndcg}, base ht: {base_ht}')
    model.save_pretrained(f'lora_checkpoints/{args.grouping_metric}/{split}/ndcg={round(base_ndcg,4)}')   

    best_ndcg = base_ndcg

    for epoch in range(2):
        running_loss = 0.0
        model.train()
        for train_batch in train_loader:
            u, seq, pos, neg = train_batch
            running_loss += train_step(model, u, seq, pos, neg, criterion, optimizer, args.device)
        # print('Epoch', epoch, 'Loss', (running_loss * 100)/len(lora_train))

    model.eval()
    test_ndcg, test_ht = 0, 0
    with torch.no_grad():
        for test_batch in test_loader:
            u, seq, pos, test_items, mask = test_batch
            batch_ndcg, batch_ht = eval_step(model, u, seq, pos, test_items, mask)
            test_ndcg += batch_ndcg
            test_ht += batch_ht
            full_ndcg += batch_ndcg
            full_ht += batch_ht
    test_ndcg /= len(lora_test)
    test_ht /= len(lora_test)
    full_len += len(lora_test)
    # print(f'test ndcg: {test_ndcg}, test ht: {test_ht}')

    if test_ndcg > best_ndcg:
        best_ndcg = test_ndcg
    
    model.save_pretrained(f'lora_checkpoints/{args.grouping_metric}/{split}/ndcg={round(best_ndcg,4)}')
    # torch.save(model.state_dict(), f'lora_checkpoints/{args.grouping_metric}/{split}/ndcg={round(best_ndcg,4)}.pth')


print(f'full ndcg: {full_ndcg / full_len}, full ht: {full_ht / full_len}')